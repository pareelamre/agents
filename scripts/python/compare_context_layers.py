from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.custom.edge_agent import (
    analyze_market_baseline,
    analyze_market_with_context_layer,
    build_market_context,
    fetch_live_markets,
    shortlist_markets,
    to_market_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline market analysis against the enhanced context-layer analysis."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="How many live markets to compare.",
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=200,
        help="How many live markets to scan from Gamma before selecting the sample.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Optional query used to focus the market sample.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name for both baseline and enhanced runs.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="Optional OpenAI-compatible base URL, for example: https://llm.scads.ai/v1",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum probability edge required before recommending a trade.",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=0.10,
        help="Maximum bankroll fraction to recommend in dry-run output.",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=3,
        help="Maximum number of news articles to enrich per market.",
    )
    parser.add_argument(
        "--article-max-chars",
        type=int,
        default=4000,
        help="Maximum article text length to pass to the LLM per article.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a readable text report.",
    )
    return parser.parse_args()


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [result for result in results if "error" not in result]
    if not completed:
        return {
            "completed_markets": 0,
            "failed_markets": len(results),
        }

    probability_deltas = [
        abs(
            result["enhanced"]["analysis"]["probability_yes"]
            - result["baseline"]["analysis"]["probability_yes"]
        )
        for result in completed
    ]
    confidence_deltas = [
        result["enhanced"]["analysis"]["confidence"]
        - result["baseline"]["analysis"]["confidence"]
        for result in completed
    ]
    recommendation_changes = sum(
        result["baseline"]["recommendation"]["action"]
        != result["enhanced"]["recommendation"]["action"]
        for result in completed
    )
    side_changes = sum(
        result["baseline"]["recommendation"]["side"]
        != result["enhanced"]["recommendation"]["side"]
        for result in completed
    )
    avg_news_articles = mean(
        len(result["enhanced"].get("context_layer", {}).get("news_articles") or [])
        for result in completed
    )

    return {
        "completed_markets": len(completed),
        "failed_markets": len(results) - len(completed),
        "recommendation_changes": recommendation_changes,
        "side_changes": side_changes,
        "avg_abs_probability_delta": round(mean(probability_deltas), 4),
        "avg_confidence_delta": round(mean(confidence_deltas), 4),
        "avg_news_articles": round(avg_news_articles, 2),
    }


def print_text_report(payload: dict[str, Any]) -> None:
    print("Context Layer Comparison")
    print(
        f"Model: {payload['model']} | Base URL: {payload['base_url'] or 'default'} | Query: {payload['query'] or 'top liquid markets'}"
    )
    summary = payload["summary"]
    print(
        "Summary: "
        f"{summary.get('completed_markets', 0)} completed, "
        f"{summary.get('failed_markets', 0)} failed, "
        f"{summary.get('recommendation_changes', 0)} recommendation changes, "
        f"avg |delta P(YES)|={summary.get('avg_abs_probability_delta', 0.0):.4f}, "
        f"avg news articles={summary.get('avg_news_articles', 0.0):.2f}"
    )
    print()

    for index, result in enumerate(payload["results"], start=1):
        if "error" in result:
            print(f"{index}. {result['market']['question']}")
            print(f"   ERROR: {result['error']}")
            continue

        baseline = result["baseline"]
        enhanced = result["enhanced"]
        baseline_rec = baseline["recommendation"]
        enhanced_rec = enhanced["recommendation"]
        print(f"{index}. {result['market']['question']}")
        print(
            "   Baseline: "
            f"P(YES)={baseline['analysis']['probability_yes']:.3f}, "
            f"conf={baseline['analysis']['confidence']:.3f}, "
            f"action={baseline_rec['action']}"
        )
        print(
            "   Enhanced: "
            f"P(YES)={enhanced['analysis']['probability_yes']:.3f}, "
            f"conf={enhanced['analysis']['confidence']:.3f}, "
            f"action={enhanced_rec['action']}, "
            f"news={len(enhanced.get('context_layer', {}).get('news_articles') or [])}"
        )
        print(
            "   Delta: "
            f"{enhanced['analysis']['probability_yes'] - baseline['analysis']['probability_yes']:+.3f} P(YES), "
            f"{enhanced['analysis']['confidence'] - baseline['analysis']['confidence']:+.3f} confidence"
        )
        if result["changed"]:
            print("   Change: recommendation changed")
        print()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key, base_url=args.base_url or None)
    raw_markets = fetch_live_markets(args.fetch_limit)
    snapshots = [
        snapshot
        for raw_market in raw_markets
        if (snapshot := to_market_snapshot(raw_market))
    ]
    sample = shortlist_markets(snapshots, args.query, args.count)

    results: list[dict[str, Any]] = []
    for market in sample:
        record: dict[str, Any] = {"market": {"id": market.id, "question": market.question}}
        try:
            baseline = analyze_market_baseline(
                client=client,
                model=args.model,
                market=market,
                query=args.query,
                min_edge=args.min_edge,
                max_size=args.max_size,
            )
            context_layer = build_market_context(
                market=market,
                query=args.query,
                news_limit=args.news_limit,
                article_max_chars=args.article_max_chars,
                client=client,
                model=args.model,
            )
            enhanced = analyze_market_with_context_layer(
                client=client,
                model=args.model,
                market=market,
                query=args.query,
                min_edge=args.min_edge,
                max_size=args.max_size,
                context_layer=context_layer,
            )
            record["baseline"] = baseline
            record["enhanced"] = enhanced
            record["changed"] = (
                baseline["recommendation"]["action"]
                != enhanced["recommendation"]["action"]
                or baseline["recommendation"]["side"]
                != enhanced["recommendation"]["side"]
            )
        except Exception as exc:  # pragma: no cover - live integration path
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)

    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "query": args.query,
        "count": len(sample),
        "news_limit": args.news_limit,
        "summary": summarize_results(results),
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text_report(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
