from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.custom.edge_agent import (
    MarketSnapshot,
    build_analysis_prompt,
    build_baseline_analysis_prompt,
    build_market_context,
    call_model,
    load_local_api_key,
    make_recommendation,
)

CLOB_MARKETS_URL = "https://clob.polymarket.com/markets"
CLOB_PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"
PAGE_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a historical backtest comparing baseline and context-layer forecasts on resolved Polymarket markets."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="How many resolved markets to backtest.",
    )
    parser.add_argument(
        "--resolved-within-days",
        type=int,
        default=20,
        help="Only use markets resolved within this many days from now.",
    )
    parser.add_argument(
        "--days-before-end",
        type=int,
        default=3,
        help="Forecast snapshot time as N days before market end.",
    )
    parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=7,
        help="How many days of news history to search before each snapshot time.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=250,
        help="Maximum recent CLOB pages to scan backward from the tail of the closed-market feed.",
    )
    parser.add_argument(
        "--include-sports",
        action="store_true",
        help="Include sports markets in the backtest universe.",
    )
    parser.add_argument(
        "--require-news",
        action="store_true",
        help="Only score markets where the historical context layer retrieved at least one news article.",
    )
    parser.add_argument(
        "--include-tags",
        default="politics,finance,earnings,geopolitics,commodities,equities,crypto",
        help="Comma-separated lowercase tag filter. Market must have at least one of these tags.",
    )
    parser.add_argument(
        "--exclude-tags",
        default="sports,games,mentions,youtube,rogan,recurring,hide from new,crypto prices,multi strikes,1h,culture,tweet markets,approval,approvals,trump daily",
        help="Comma-separated lowercase tags to exclude from the backtest universe.",
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
        default=2,
        help="Maximum number of news articles to enrich per market.",
    )
    parser.add_argument(
        "--article-max-chars",
        type=int,
        default=3000,
        help="Maximum article text length to pass to the LLM per article.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a readable text report.",
    )
    return parser.parse_args()


def request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    request_url = url
    if params:
        request_url = f"{url}?{urlencode(params)}"
    request = Request(request_url, headers={"User-Agent": "polymarket-edge-agent"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def encode_cursor(offset: int) -> str:
    return base64.b64encode(str(max(offset, 0)).encode("utf-8")).decode("utf-8")


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_csv_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def clamp_probability(value: float, epsilon: float = 1e-6) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def market_group_key(question: str) -> str:
    normalized = question.lower()
    normalized = normalized.replace('"', "")
    normalized = normalized.replace("'", "")
    normalized = re_sub_many(
        normalized,
        [
            (r"identify .* as satoshi", "identify # as satoshi"),
            (r"post .* tweets", "post # tweets"),
            (r"post .* truth social posts", "post # truth social posts"),
            (r"above \$?[0-9,]+(?:\.[0-9]+)?", "above $#"),
            (r"between \$?[0-9,]+(?:\.[0-9]+)? and \$?[0-9,]+(?:\.[0-9]+)?", "between $# and $#"),
            (r"greater than \$?[0-9,]+(?:\.[0-9]+)?", "greater than $#"),
            (r"less than \$?[0-9,]+(?:\.[0-9]+)?", "less than $#"),
            (r"over [0-9,]+(?:\.[0-9]+)?", "over #"),
        ],
    )
    normalized = re_sub_many(
        normalized,
        [
            (r"\b\d{4}-\d{2}-\d{2}\b", "#"),
            (r"\b[a-z]{3,9}\s+\d{1,2}\b", "#"),
            (r"[<>]?\$?\d[\d,]*(?:\.\d+)?(?:-\$?\d[\d,]*(?:\.\d+)?)?\+?", "#"),
        ],
    )
    normalized = " ".join(normalized.split())
    return normalized


def re_sub_many(text: str, replacements: list[tuple[str, str]]) -> str:
    import re

    updated = text
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated)
    return updated


def fetch_market_page(page_index: int) -> list[dict[str, Any]]:
    payload = request_json(
        CLOB_MARKETS_URL,
        {"closed": "true", "next_cursor": encode_cursor(page_index * PAGE_SIZE)},
    )
    return payload.get("data") or []


def page_date_range(page: list[dict[str, Any]]) -> tuple[datetime | None, datetime | None]:
    dates = [
        dt
        for item in page
        if (dt := parse_iso_datetime(item.get("end_date_iso")))
    ]
    if not dates:
        return (None, None)
    return (dates[0], dates[-1])


def find_last_nonempty_page(max_page_hint: int = 1) -> int:
    low = 0
    high = max_page_hint
    while fetch_market_page(high):
        low = high
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        if fetch_market_page(mid):
            low = mid
        else:
            high = mid
    return low


def find_start_page(target_start: datetime, last_page: int) -> int:
    low = 0
    high = last_page
    while low < high:
        mid = (low + high) // 2
        _, page_end = page_date_range(fetch_market_page(mid))
        if page_end is None or page_end < target_start:
            low = mid + 1
        else:
            high = mid
    return low


def extract_yes_no_tokens(tokens: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    yes_token = next(
        (token for token in tokens if str(token.get("outcome", "")).strip().lower() == "yes"),
        None,
    )
    no_token = next(
        (token for token in tokens if str(token.get("outcome", "")).strip().lower() == "no"),
        None,
    )
    if yes_token is None or no_token is None:
        return None
    return (yes_token, no_token)


def fetch_historical_yes_price(token_id: str, snapshot_time: datetime) -> float | None:
    snapshot_ts = int(snapshot_time.timestamp())
    start_ts = int((snapshot_time - timedelta(days=7)).timestamp())
    try:
        history_payload = request_json(
            CLOB_PRICE_HISTORY_URL,
            {
                "market": token_id,
                "startTs": start_ts,
                "endTs": snapshot_ts,
                "fidelity": 60,
            },
        )
    except Exception:
        return None
    history = history_payload.get("history") or []
    if history:
        return float(history[-1]["p"])
    return None


def make_historical_snapshot(market: dict[str, Any], yes_price: float) -> MarketSnapshot:
    no_price = 1.0 - yes_price
    return MarketSnapshot(
        id=str(market.get("condition_id", "")),
        question=str(market.get("question", "")).strip(),
        description=str(market.get("description", "")).strip(),
        slug=str(market.get("market_slug", "")).strip(),
        event_slug="",
        end_date=str(market.get("end_date_iso", "")).strip(),
        liquidity=0.0,
        volume=0.0,
        volume_24h=0.0,
        spread=0.0,
        best_bid=None,
        best_ask=None,
        yes_price=yes_price,
        no_price=no_price,
        outcomes=["Yes", "No"],
        outcome_prices=[yes_price, no_price],
        restricted=False,
        url=f"https://polymarket.com/event/{market.get('market_slug', '')}",
        context="",
    )


def with_snapshot_instructions(prompt: str, snapshot_time: datetime) -> str:
    return (
        f"{prompt}\n"
        "Historical evaluation instructions:\n"
        f"- Treat the current time as {snapshot_time.isoformat()}.\n"
        "- Use only information that would have been publicly available by that time.\n"
        "- Do not assume knowledge of the final market outcome.\n"
    )


def analyze_historical_baseline(
    client: OpenAI,
    model: str,
    market: MarketSnapshot,
    snapshot_time: datetime,
    min_edge: float,
    max_size: float,
) -> dict[str, Any]:
    analysis = call_model(
        client=client,
        model=model,
        prompt=with_snapshot_instructions(
            build_baseline_analysis_prompt("", market), snapshot_time
        ),
        system_prompt="You are a disciplined prediction-market analyst. Be concise and probabilistic.",
    )
    recommendation = make_recommendation(
        market, analysis, min_edge=min_edge, max_size=max_size
    )
    return {
        "analysis": analysis,
        "recommendation": asdict(recommendation),
        "context_layer": {"resolution_criteria": "", "news_articles": []},
    }


def analyze_historical_enhanced(
    client: OpenAI,
    model: str,
    market: MarketSnapshot,
    snapshot_time: datetime,
    context_layer: Any,
    min_edge: float,
    max_size: float,
) -> dict[str, Any]:
    analysis = call_model(
        client=client,
        model=model,
        prompt=with_snapshot_instructions(
            build_analysis_prompt("", market, context_layer), snapshot_time
        ),
        system_prompt="You are a disciplined prediction-market analyst. Be concise and probabilistic.",
    )
    recommendation = make_recommendation(
        market, analysis, min_edge=min_edge, max_size=max_size
    )
    return {
        "analysis": analysis,
        "recommendation": asdict(recommendation),
        "context_layer": asdict(context_layer),
    }


def compute_trade_pnl(action: str, market_price: float, outcome_yes: int) -> float:
    if action == "BUY_YES":
        return outcome_yes - market_price
    if action == "BUY_NO":
        return (1 - outcome_yes) - market_price
    return 0.0


def compute_metrics(result: dict[str, Any], outcome_yes: int) -> dict[str, float]:
    probability_yes = clamp_probability(
        float(result["analysis"].get("probability_yes", 0.5))
    )
    recommendation = result["recommendation"]
    action = recommendation["action"]
    market_price = float(recommendation["market_price"])
    size_fraction = float(recommendation["size_fraction"])
    brier = (probability_yes - outcome_yes) ** 2
    log_loss = -(
        outcome_yes * math.log(probability_yes)
        + (1 - outcome_yes) * math.log(1 - probability_yes)
    )
    pnl_per_contract = compute_trade_pnl(action, market_price, outcome_yes)
    sized_pnl = pnl_per_contract * size_fraction
    direction = 1 if probability_yes >= 0.5 else 0
    return {
        "brier": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "directional_accuracy": float(direction == outcome_yes),
        "pnl_per_contract": round(pnl_per_contract, 6),
        "sized_pnl": round(sized_pnl, 6),
    }


def summarize_side(results: list[dict[str, Any]], side: str) -> dict[str, Any]:
    completed = [result for result in results if "error" not in result]
    if not completed:
        return {}

    metrics = [result[side]["metrics"] for result in completed]
    return {
        "avg_brier": round(mean(metric["brier"] for metric in metrics), 6),
        "avg_log_loss": round(mean(metric["log_loss"] for metric in metrics), 6),
        "directional_accuracy": round(
            mean(metric["directional_accuracy"] for metric in metrics), 6
        ),
        "avg_pnl_per_contract": round(
            mean(metric["pnl_per_contract"] for metric in metrics), 6
        ),
        "avg_sized_pnl": round(mean(metric["sized_pnl"] for metric in metrics), 6),
        "buy_signals": sum(
            result[side]["recommendation"]["action"] != "PASS" for result in completed
        ),
    }


def build_backtest_universe(
    count: int,
    resolved_within_days: int,
    days_before_end: int,
    news_lookback_days: int,
    max_pages: int,
    include_sports: bool,
    include_tags: set[str],
    exclude_tags: set[str],
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    latest_snapshot = now - timedelta(days=1)
    earliest_end = now - timedelta(days=resolved_within_days)
    earliest_snapshot = now - timedelta(days=30)

    last_page = find_last_nonempty_page()
    raw_candidates: list[dict[str, Any]] = []
    seen_conditions: set[str] = set()
    first_page = max(0, last_page - max_pages + 1)
    page_indices = list(range(last_page, first_page - 1, -1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        page_results = executor.map(fetch_market_page, page_indices)
        for page_index, page in zip(page_indices, page_results):
            for market in page:
                condition_id = str(market.get("condition_id", "")).strip()
                if not condition_id or condition_id in seen_conditions:
                    continue
                seen_conditions.add(condition_id)
                end_time = parse_iso_datetime(market.get("end_date_iso"))
                if end_time is None or end_time > now or end_time < earliest_end:
                    continue
                tags = {str(tag).strip().lower() for tag in (market.get("tags") or [])}
                if exclude_tags & tags:
                    continue
                if include_tags and not (include_tags & tags):
                    continue
                if not include_sports and {"sports", "games"} & tags:
                    continue
                question_text = str(market.get("question", "")).strip().lower()
                if {"mentions", "youtube", "rogan"} & tags:
                    continue
                if "be said during the first episode of the joe rogan experience" in question_text:
                    continue
                raw_candidates.append(market)

    raw_candidates.sort(
        key=lambda market: parse_iso_datetime(market.get("end_date_iso")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for market in raw_candidates:
        end_time = parse_iso_datetime(market.get("end_date_iso"))
        if end_time is None:
            continue
        if end_time > now:
            continue
        if end_time < earliest_end:
            break

        snapshot_time = end_time - timedelta(days=days_before_end)
        if snapshot_time > latest_snapshot or snapshot_time < earliest_snapshot:
            continue

        tokens = market.get("tokens") or []
        yes_no_tokens = extract_yes_no_tokens(tokens)
        if yes_no_tokens is None:
            continue
        yes_token, no_token = yes_no_tokens
        if not isinstance(yes_token.get("winner"), bool) or not isinstance(
            no_token.get("winner"), bool
        ):
            continue
        if yes_token["winner"] == no_token["winner"]:
            continue
        group_key = market_group_key(str(market.get("question", "")).strip())
        if group_key in seen_groups:
            continue

        yes_price = fetch_historical_yes_price(str(yes_token["token_id"]), snapshot_time)
        if yes_price is None:
            continue

        selected.append(
            {
                "market": market,
                "snapshot_time": snapshot_time,
                "outcome_yes": 1 if yes_token["winner"] else 0,
                "historical_snapshot": make_historical_snapshot(market, yes_price),
            }
        )
        seen_groups.add(group_key)
        if len(selected) >= count:
            return selected
    return selected


def print_text_report(payload: dict[str, Any]) -> None:
    print("Historical Context Layer Backtest")
    print(
        f"Model: {payload['model']} | Base URL: {payload['base_url'] or 'default'} | Snapshot rule: {payload['days_before_end']} days before end"
    )
    print(
        f"Universe: {payload['count']} markets resolved within {payload['resolved_within_days']} days | News lookback: {payload['news_lookback_days']} days"
    )
    print()
    print("Baseline summary:")
    for key, value in payload["summary"]["baseline"].items():
        print(f"- {key}: {value}")
    print("Enhanced summary:")
    for key, value in payload["summary"]["enhanced"].items():
        print(f"- {key}: {value}")
    print()

    for index, result in enumerate(payload["results"], start=1):
        if "error" in result:
            print(f"{index}. {result['market']['question']}")
            print(f"   ERROR: {result['error']}")
            continue

        print(f"{index}. {result['market']['question']}")
        print(
            f"   Outcome: {'YES' if result['outcome_yes'] else 'NO'} | Snapshot: {result['snapshot_time']}"
        )
        print(
            "   Baseline: "
            f"P(YES)={result['baseline']['analysis']['probability_yes']:.3f}, "
            f"action={result['baseline']['recommendation']['action']}, "
            f"brier={result['baseline']['metrics']['brier']:.4f}, "
            f"pnl={result['baseline']['metrics']['pnl_per_contract']:+.4f}"
        )
        print(
            "   Enhanced: "
            f"P(YES)={result['enhanced']['analysis']['probability_yes']:.3f}, "
            f"action={result['enhanced']['recommendation']['action']}, "
            f"brier={result['enhanced']['metrics']['brier']:.4f}, "
            f"pnl={result['enhanced']['metrics']['pnl_per_contract']:+.4f}, "
            f"news={len(result['enhanced']['context_layer'].get('news_articles') or [])}"
        )
        print()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.resolved_within_days + args.days_before_end + args.news_lookback_days > 30:
        raise SystemExit(
            "With the current NewsAPI developer archive limit, resolved-within-days + days-before-end + news-lookback-days must be <= 30."
        )

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if args.base_url and "llm.scads.ai" in args.base_url.lower():
        scads_api_key = load_local_api_key("SCADSAI_API_KEY", "SCADSAI_API_KEY.txt")
        openai_api_key = scads_api_key or openai_api_key
    if not openai_api_key:
        raise SystemExit(
            "No model API key found. Set OPENAI_API_KEY, or use SCADSAI_API_KEY.txt with the ScaDS base URL."
        )
    if not load_local_api_key("NEWSAPI_API_KEY", "NEWSAPI_api_key.txt"):
        raise SystemExit("NEWSAPI_API_KEY is not set and NEWSAPI_api_key.txt was not found.")

    client = OpenAI(api_key=openai_api_key, base_url=args.base_url or None)
    include_tags = parse_csv_set(args.include_tags)
    exclude_tags = parse_csv_set(args.exclude_tags)
    universe = build_backtest_universe(
        count=args.count * (5 if args.require_news else 1),
        resolved_within_days=args.resolved_within_days,
        days_before_end=args.days_before_end,
        news_lookback_days=args.news_lookback_days,
        max_pages=args.max_pages,
        include_sports=args.include_sports,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )

    results: list[dict[str, Any]] = []
    for item in universe:
        market_data = item["market"]
        market = item["historical_snapshot"]
        snapshot_time = item["snapshot_time"]
        record: dict[str, Any] = {
            "market": {
                "condition_id": market_data.get("condition_id"),
                "question": market.question,
                "end_date": market.end_date,
                "url": market.url,
                "historical_yes_price": round(market.yes_price, 4),
                "historical_no_price": round(market.no_price, 4),
            },
            "snapshot_time": snapshot_time.isoformat(),
            "outcome_yes": item["outcome_yes"],
        }
        try:
            baseline = analyze_historical_baseline(
                client=client,
                model=args.model,
                market=market,
                snapshot_time=snapshot_time,
                min_edge=args.min_edge,
                max_size=args.max_size,
            )
            context_layer = build_market_context(
                market=market,
                query="",
                news_limit=args.news_limit,
                article_max_chars=args.article_max_chars,
                client=client,
                model=args.model,
                as_of=snapshot_time,
                news_lookback_days=args.news_lookback_days,
            )
            if args.require_news and not context_layer.news_articles:
                continue
            enhanced = analyze_historical_enhanced(
                client=client,
                model=args.model,
                market=market,
                snapshot_time=snapshot_time,
                context_layer=context_layer,
                min_edge=args.min_edge,
                max_size=args.max_size,
            )
            baseline["metrics"] = compute_metrics(baseline, item["outcome_yes"])
            enhanced["metrics"] = compute_metrics(enhanced, item["outcome_yes"])
            record["baseline"] = baseline
            record["enhanced"] = enhanced
        except Exception as exc:  # pragma: no cover - live integration path
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
        if len(results) >= args.count:
            break

    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "count": len(results),
        "resolved_within_days": args.resolved_within_days,
        "days_before_end": args.days_before_end,
        "news_lookback_days": args.news_lookback_days,
        "include_sports": args.include_sports,
        "require_news": args.require_news,
        "include_tags": sorted(include_tags),
        "exclude_tags": sorted(exclude_tags),
        "summary": {
            "baseline": summarize_side(results, "baseline"),
            "enhanced": summarize_side(results, "enhanced"),
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text_report(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
