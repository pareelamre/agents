from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for the lightweight setup
    def load_dotenv() -> bool:
        return False

from openai import OpenAI


GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class MarketSnapshot:
    id: str
    question: str
    description: str
    slug: str
    event_slug: str
    end_date: str
    liquidity: float
    volume: float
    volume_24h: float
    spread: float
    best_bid: float | None
    best_ask: float | None
    yes_price: float
    no_price: float
    outcomes: list[str]
    outcome_prices: list[float]
    restricted: bool
    url: str
    context: str


@dataclass
class TradeRecommendation:
    action: str
    side: str | None
    market_price: float
    model_probability: float
    edge: float
    size_fraction: float
    confidence: float
    rationale: str


@dataclass
class ForecastingSummary:
    predicted_future_event: str
    predicted_attributes_of_future_event: list[str]
    genre_of_statement: str
    credibility_of_source: str
    conditions: list[str]
    rationale: str
    modality: str


@dataclass
class NewsArticleContext:
    url: str
    publish_time: str
    title: str
    text: str
    summary: str
    summary_llm: str
    forecasting_summaries: ForecastingSummary


@dataclass
class MarketContextLayer:
    resolution_criteria: str
    news_articles: list[NewsArticleContext]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run Polymarket agent that finds a market, gathers context, and proposes a trade."
    )
    parser.add_argument(
        "--query",
        default="",
        help="Search phrase for markets to analyze, for example: 'fed cuts' or 'ukraine ceasefire'.",
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=150,
        help="How many live markets to scan from Gamma.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
        help="How many candidate markets to send to the model.",
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
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="Optional OpenAI-compatible base URL, for example: https://llm.scads.ai/v1",
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
        "--list-only",
        action="store_true",
        help="List matched candidate markets without calling the LLM.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a readable text report.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_text(text: str, max_chars: int) -> str:
    normalized = clean_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def parse_jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


def load_local_api_key(env_var: str, filename: str) -> str:
    existing = clean_text(os.getenv(env_var, ""))
    if existing:
        return existing

    candidate = REPO_ROOT / filename
    if not candidate.exists():
        return ""

    value = clean_text(candidate.read_text(encoding="utf-8"))
    if value:
        os.environ[env_var] = value
    return value


def request_json(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    query = urlencode(params)
    request = Request(f"{url}?{query}", headers={"User-Agent": "polymarket-edge-agent"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_live_markets(fetch_limit: int) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    offset = 0
    page_size = min(fetch_limit, 100)

    while len(markets) < fetch_limit:
        batch = request_json(
            GAMMA_MARKETS_URL,
            {
                "active": "true",
                "closed": "false",
                "archived": "false",
                "limit": page_size,
                "offset": offset,
            },
        )
        if not batch:
            break
        markets.extend(batch)
        if len(batch) < page_size:
            break
        offset += len(batch)

    return markets[:fetch_limit]


def extract_resolution_criteria(description: str) -> str:
    paragraphs = [
        segment.strip() for segment in re.split(r"\n\s*\n", description) if segment.strip()
    ]
    if not paragraphs:
        return ""

    keywords = (
        "resolve",
        "resolution",
        "count",
        "qualify",
        "otherwise",
        "official",
        "source",
        "unless",
    )
    selected = [
        clean_text(paragraph)
        for paragraph in paragraphs
        if any(keyword in paragraph.lower() for keyword in keywords)
    ]
    if not selected:
        selected = [clean_text(paragraphs[0])]

    return "\n\n".join(dict.fromkeys(selected))


def build_news_query(market: MarketSnapshot, query: str) -> str:
    cleaned_query = clean_text(query)
    if cleaned_query:
        return cleaned_query

    stopwords = {
        "will",
        "what",
        "when",
        "where",
        "which",
        "before",
        "after",
        "happen",
        "happens",
        "market",
    }
    filtered_terms = [
        term for term in tokenize(market.question) if term not in stopwords
    ]
    ordered_terms = list(dict.fromkeys(filtered_terms))
    return " ".join(ordered_terms[:8]) or market.question


def extract_article_with_newspaper(url: str) -> tuple[str, str]:
    try:
        from newspaper import Article as NewspaperArticle
    except ImportError:
        return ("", "")

    article = NewspaperArticle(url=url, language="en")
    article.download()
    article.parse()

    extracted_text = clean_text(article.text)
    extracted_summary = ""
    try:
        article.nlp()
        extracted_summary = clean_text(article.summary)
    except Exception:
        # newspaper3k summary generation may fail when local NLP data is missing.
        extracted_summary = truncate_text(extracted_text, 800)

    return (extracted_text, extracted_summary)


def normalize_forecasting_summary(data: dict[str, Any] | list[Any]) -> ForecastingSummary:
    if isinstance(data, list):
        data = next(
            (item for item in data if isinstance(item, dict)),
            {},
        )
    if not isinstance(data, dict):
        data = {}

    return ForecastingSummary(
        predicted_future_event=clean_text(data.get("predicted_future_event", "")),
        predicted_attributes_of_future_event=[
            clean_text(item)
            for item in (data.get("predicted_attributes_of_future_event") or [])
            if clean_text(item)
        ],
        genre_of_statement=clean_text(data.get("genre_of_statement", "")),
        credibility_of_source=clean_text(data.get("credibility_of_source", "")),
        conditions=[
            clean_text(item) for item in (data.get("conditions") or []) if clean_text(item)
        ],
        rationale=clean_text(data.get("rationale", "")),
        modality=clean_text(data.get("modality", "")),
    )


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def call_model(client: OpenAI, model: str, prompt: str, system_prompt: str) -> dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
    except Exception:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt + " Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"

    return json.loads(strip_code_fences(content))


def analyze_article_with_llm(
    client: OpenAI,
    model: str,
    market: MarketSnapshot,
    resolution_criteria: str,
    article: dict[str, str],
    as_of: datetime | None = None,
) -> tuple[str, ForecastingSummary]:
    as_of_text = (
        f"Historical snapshot time: {as_of.isoformat()}\n"
        if as_of is not None
        else ""
    )
    prompt = (
        "Analyze this article in the context of a Polymarket contract.\n"
        "Return strict JSON with keys: summary_llm, forecasting_summaries.\n"
        "forecasting_summaries must contain keys: predicted_future_event, predicted_attributes_of_future_event, "
        "genre_of_statement, credibility_of_source, conditions, rationale, modality.\n"
        "Use short strings and arrays. Do not include markdown fences.\n\n"
        f"{as_of_text}"
        f"Market question: {market.question}\n"
        f"Market description: {market.description}\n"
        f"Resolution criteria: {resolution_criteria}\n"
        f"Article title: {article['title']}\n"
        f"Article publish time: {article['publish_time']}\n"
        f"Article text: {article['text']}\n"
    )
    response = call_model(
        client=client,
        model=model,
        prompt=prompt,
        system_prompt=(
            "You are a forecasting research analyst. Extract forward-looking claims, conditions, and source credibility."
        ),
    )
    return (
        clean_text(response.get("summary_llm", "")),
        normalize_forecasting_summary(response.get("forecasting_summaries") or {}),
    )


def collect_news_articles(
    market: MarketSnapshot,
    query: str,
    limit: int,
    max_chars: int,
    client: OpenAI | None,
    model: str,
    resolution_criteria: str,
    as_of: datetime | None = None,
    news_lookback_days: int = 30,
) -> list[NewsArticleContext]:
    if limit <= 0:
        return []

    if not load_local_api_key("NEWSAPI_API_KEY", "NEWSAPI_api_key.txt"):
        return []

    from agents.connectors.news import News

    news_client = News()
    search_query = build_news_query(market, query)
    as_of = as_of or datetime.now(timezone.utc)
    start_dt = as_of - timedelta(days=max(news_lookback_days, 1))
    start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_date = as_of.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        response = news_client.API.get_everything(
            q=search_query,
            language=news_client.configs["language"],
            sort_by="publishedAt",
            page_size=min(limit, 100),
            from_param=start_date,
            to=end_date,
        )
    except Exception:
        return []

    articles: list[NewsArticleContext] = []
    seen_urls: set[str] = set()

    for raw_article in response.get("articles", []):
        url = clean_text(raw_article.get("url", ""))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        title = clean_text(raw_article.get("title", ""))
        publish_time = clean_text(raw_article.get("publishedAt", ""))
        text = clean_text(raw_article.get("content", ""))
        summary = ""
        if publish_time:
            try:
                published_at_dt = datetime.fromisoformat(
                    publish_time.replace("Z", "+00:00")
                )
                if published_at_dt > as_of or published_at_dt < start_dt:
                    continue
            except ValueError:
                pass

        try:
            newspaper_text, newspaper_summary = extract_article_with_newspaper(url)
            if newspaper_text:
                text = newspaper_text
            summary = newspaper_summary
        except Exception:
            summary = ""

        text = truncate_text(text, max_chars)
        summary = truncate_text(summary, 1200)

        summary_llm = ""
        forecasting_summaries = normalize_forecasting_summary({})
        if client is not None:
            summary_llm, forecasting_summaries = analyze_article_with_llm(
                client=client,
                model=model,
                market=market,
                resolution_criteria=resolution_criteria,
                article={
                    "title": title,
                    "publish_time": publish_time,
                    "text": text,
                },
                as_of=as_of,
            )

        articles.append(
            NewsArticleContext(
                url=url,
                publish_time=publish_time,
                title=title,
                text=text,
                summary=summary,
                summary_llm=summary_llm,
                forecasting_summaries=forecasting_summaries,
            )
        )

        if len(articles) >= limit:
            break

    return articles


def build_market_context(
    market: MarketSnapshot,
    query: str,
    news_limit: int,
    article_max_chars: int,
    client: OpenAI | None,
    model: str,
    as_of: datetime | None = None,
    news_lookback_days: int = 30,
) -> MarketContextLayer:
    resolution_criteria = extract_resolution_criteria(market.description)
    news_articles = collect_news_articles(
        market=market,
        query=query,
        limit=news_limit,
        max_chars=article_max_chars,
        client=client,
        model=model,
        resolution_criteria=resolution_criteria,
        as_of=as_of,
        news_lookback_days=news_lookback_days,
    )
    return MarketContextLayer(
        resolution_criteria=resolution_criteria,
        news_articles=news_articles,
    )


def to_market_snapshot(raw_market: dict[str, Any]) -> MarketSnapshot | None:
    outcomes = parse_jsonish_list(raw_market.get("outcomes"))
    prices = [
        safe_float(price) for price in parse_jsonish_list(raw_market.get("outcomePrices"))
    ]

    if len(outcomes) != 2 or len(prices) != 2:
        return None

    normalized_outcomes = [str(outcome).strip().lower() for outcome in outcomes]
    if "yes" not in normalized_outcomes or "no" not in normalized_outcomes:
        return None

    yes_index = normalized_outcomes.index("yes")
    no_index = normalized_outcomes.index("no")

    event_slug = ""
    context = ""
    events = raw_market.get("events") or []
    if events:
        event = events[0] or {}
        event_slug = str(event.get("slug", "")).strip()
        context = str(
            (event.get("eventMetadata") or {}).get("context_description", "")
        ).strip()

    slug = str(raw_market.get("slug", "")).strip()
    url_slug = event_slug or slug

    return MarketSnapshot(
        id=str(raw_market.get("id", "")),
        question=str(raw_market.get("question", "")).strip(),
        description=str(raw_market.get("description", "")).strip(),
        slug=slug,
        event_slug=event_slug,
        end_date=str(raw_market.get("endDate", "")).strip(),
        liquidity=safe_float(raw_market.get("liquidityNum") or raw_market.get("liquidity")),
        volume=safe_float(raw_market.get("volumeNum") or raw_market.get("volume")),
        volume_24h=safe_float(raw_market.get("volume24hr")),
        spread=safe_float(raw_market.get("spread")),
        best_bid=(
            safe_float(raw_market.get("bestBid"), default=-1.0)
            if raw_market.get("bestBid") is not None
            else None
        ),
        best_ask=(
            safe_float(raw_market.get("bestAsk"), default=-1.0)
            if raw_market.get("bestAsk") is not None
            else None
        ),
        yes_price=prices[yes_index],
        no_price=prices[no_index],
        outcomes=[str(outcome) for outcome in outcomes],
        outcome_prices=prices,
        restricted=bool(raw_market.get("restricted", False)),
        url=f"https://polymarket.com/event/{url_slug}"
        if url_slug
        else "https://polymarket.com",
        context=context,
    )


def rank_market(snapshot: MarketSnapshot, query: str) -> tuple[float, float, float]:
    if not query.strip():
        return (0.0, snapshot.liquidity, snapshot.volume_24h)

    terms = tokenize(query)
    haystack = f"{snapshot.question} {snapshot.description} {snapshot.context}".lower()
    lexical_hits = sum(haystack.count(term) for term in terms)
    return (float(lexical_hits), snapshot.liquidity, snapshot.volume_24h)


def shortlist_markets(
    markets: list[MarketSnapshot], query: str, candidates: int
) -> list[MarketSnapshot]:
    ranked = sorted(markets, key=lambda market: rank_market(market, query), reverse=True)
    if query.strip():
        matched = [market for market in ranked if rank_market(market, query)[0] > 0]
        return (matched or ranked)[:candidates]
    return ranked[:candidates]


def build_analysis_prompt(
    query: str, market: MarketSnapshot, context_layer: MarketContextLayer
) -> str:
    focus = query.strip() or "Find the best dry-run trade in this market."
    serialized_context = json.dumps(asdict(context_layer), indent=2)
    return (
        "Evaluate this binary Polymarket contract and estimate the fair probability of YES.\n"
        "Return strict JSON with keys: probability_yes, confidence, rationale, key_drivers, risks.\n"
        "Rules:\n"
        "- probability_yes and confidence must be numbers between 0 and 1.\n"
        "- key_drivers and risks must be arrays of short strings.\n"
        "- Keep rationale under 100 words.\n"
        "- Use the context layer and resolution criteria explicitly.\n"
        "- Do not include markdown fences.\n\n"
        f"User focus: {focus}\n"
        f"Question: {market.question}\n"
        f"Description: {market.description}\n"
        f"Market end date: {market.end_date}\n"
        f"Current YES price: {market.yes_price:.3f}\n"
        f"Current NO price: {market.no_price:.3f}\n"
        f"Liquidity: {market.liquidity:.2f}\n"
        f"24h volume: {market.volume_24h:.2f}\n"
        f"Additional market context: {market.context or 'None'}\n"
        f"Context layer:\n{serialized_context}\n"
    )


def build_baseline_analysis_prompt(query: str, market: MarketSnapshot) -> str:
    focus = query.strip() or "Find the best dry-run trade in this market."
    return (
        "Evaluate this binary Polymarket contract and estimate the fair probability of YES.\n"
        "Return strict JSON with keys: probability_yes, confidence, rationale, key_drivers, risks.\n"
        "Rules:\n"
        "- probability_yes and confidence must be numbers between 0 and 1.\n"
        "- key_drivers and risks must be arrays of short strings.\n"
        "- Keep rationale under 100 words.\n"
        "- Use the market question and description carefully.\n"
        "- Do not include markdown fences.\n\n"
        f"User focus: {focus}\n"
        f"Question: {market.question}\n"
        f"Description: {market.description}\n"
        f"Market end date: {market.end_date}\n"
        f"Current YES price: {market.yes_price:.3f}\n"
        f"Current NO price: {market.no_price:.3f}\n"
        f"Liquidity: {market.liquidity:.2f}\n"
        f"24h volume: {market.volume_24h:.2f}\n"
        f"Additional market context: {market.context or 'None'}\n"
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def make_recommendation(
    market: MarketSnapshot,
    analysis: dict[str, Any],
    min_edge: float,
    max_size: float,
) -> TradeRecommendation:
    probability_yes = clamp(safe_float(analysis.get("probability_yes")), 0.0, 1.0)
    confidence = clamp(safe_float(analysis.get("confidence"), 0.5), 0.0, 1.0)
    yes_edge = probability_yes - market.yes_price
    no_edge = (1.0 - probability_yes) - market.no_price

    action = "PASS"
    side = None
    market_price = market.yes_price
    chosen_edge = max(yes_edge, no_edge)

    if yes_edge >= no_edge and yes_edge >= min_edge:
        action = "BUY_YES"
        side = "YES"
        market_price = market.yes_price
        chosen_edge = yes_edge
    elif no_edge > yes_edge and no_edge >= min_edge:
        action = "BUY_NO"
        side = "NO"
        market_price = market.no_price
        chosen_edge = no_edge

    raw_size = chosen_edge * confidence * 1.5
    size_fraction = clamp(raw_size, 0.0, max_size) if action != "PASS" else 0.0

    rationale = str(analysis.get("rationale", "")).strip()
    return TradeRecommendation(
        action=action,
        side=side,
        market_price=round(market_price, 4),
        model_probability=round(probability_yes, 4),
        edge=round(chosen_edge, 4),
        size_fraction=round(size_fraction, 4),
        confidence=round(confidence, 4),
        rationale=rationale,
    )


def candidate_payload(market: MarketSnapshot) -> dict[str, Any]:
    return {
        "id": market.id,
        "question": market.question,
        "yes_price": round(market.yes_price, 4),
        "no_price": round(market.no_price, 4),
        "liquidity": round(market.liquidity, 2),
        "volume_24h": round(market.volume_24h, 2),
        "restricted": market.restricted,
        "url": market.url,
    }


def analyze_market(
    client: OpenAI,
    model: str,
    market: MarketSnapshot,
    query: str,
    min_edge: float,
    max_size: float,
    news_limit: int,
    article_max_chars: int,
) -> dict[str, Any]:
    context_layer = build_market_context(
        market=market,
        query=query,
        news_limit=news_limit,
        article_max_chars=article_max_chars,
        client=client,
        model=model,
    )
    return analyze_market_with_context_layer(
        client=client,
        model=model,
        market=market,
        query=query,
        min_edge=min_edge,
        max_size=max_size,
        context_layer=context_layer,
    )


def analyze_market_with_context_layer(
    client: OpenAI,
    model: str,
    market: MarketSnapshot,
    query: str,
    min_edge: float,
    max_size: float,
    context_layer: MarketContextLayer,
) -> dict[str, Any]:
    analysis = call_model(
        client=client,
        model=model,
        prompt=build_analysis_prompt(query, market, context_layer),
        system_prompt="You are a disciplined prediction-market analyst. Be concise and probabilistic.",
    )
    recommendation = make_recommendation(
        market, analysis, min_edge=min_edge, max_size=max_size
    )
    return {
        "market": candidate_payload(market),
        "context_layer": asdict(context_layer),
        "analysis": analysis,
        "recommendation": asdict(recommendation),
    }


def analyze_market_baseline(
    client: OpenAI,
    model: str,
    market: MarketSnapshot,
    query: str,
    min_edge: float,
    max_size: float,
) -> dict[str, Any]:
    analysis = call_model(
        client=client,
        model=model,
        prompt=build_baseline_analysis_prompt(query, market),
        system_prompt="You are a disciplined prediction-market analyst. Be concise and probabilistic.",
    )
    recommendation = make_recommendation(
        market, analysis, min_edge=min_edge, max_size=max_size
    )
    return {
        "market": candidate_payload(market),
        "context_layer": asdict(MarketContextLayer(resolution_criteria="", news_articles=[])),
        "analysis": analysis,
        "recommendation": asdict(recommendation),
    }


def print_text_report(result: dict[str, Any]) -> None:
    market = result["market"]
    recommendation = result["recommendation"]
    analysis = result["analysis"]
    context_layer = result.get("context_layer") or {}

    print(f"Market: {market['question']}")
    print(f"URL: {market['url']}")
    print(
        f"Prices: YES {market['yes_price']:.3f} / NO {market['no_price']:.3f} | Liquidity {market['liquidity']:.2f} | 24h volume {market['volume_24h']:.2f}"
    )
    print(
        f"Model: P(YES)={recommendation['model_probability']:.3f} | confidence={recommendation['confidence']:.3f} | edge={recommendation['edge']:.3f}"
    )
    print(
        f"Action: {recommendation['action']} | size_fraction={recommendation['size_fraction']:.3f}"
    )
    print(f"Rationale: {recommendation['rationale']}")

    resolution_criteria = clean_text(context_layer.get("resolution_criteria", ""))
    if resolution_criteria:
        print("Resolution criteria:")
        print(resolution_criteria)

    news_articles = context_layer.get("news_articles") or []
    if news_articles:
        print(f"News articles analyzed: {len(news_articles)}")

    drivers = analysis.get("key_drivers") or []
    if drivers:
        print("Key drivers:")
        for driver in drivers:
            print(f"- {driver}")

    risks = analysis.get("risks") or []
    if risks:
        print("Risks:")
        for risk in risks:
            print(f"- {risk}")


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        raw_markets = fetch_live_markets(args.fetch_limit)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Failed to fetch Polymarket markets: {exc}", file=sys.stderr)
        return 1

    snapshots = [
        snapshot
        for raw_market in raw_markets
        if (snapshot := to_market_snapshot(raw_market))
    ]
    candidates = shortlist_markets(snapshots, args.query, args.candidates)

    if not candidates:
        print("No live binary markets matched the current filter.", file=sys.stderr)
        return 1

    if args.list_only:
        payload = [candidate_payload(candidate) for candidate in candidates]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for candidate in payload:
                print(
                    f"{candidate['question']} | YES {candidate['yes_price']:.3f} | NO {candidate['no_price']:.3f} | liquidity {candidate['liquidity']:.2f}"
                )
                print(candidate["url"])
        return 0

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY is not set. Run with --list-only to inspect markets, or add the key to your environment/.env.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=api_key, base_url=args.base_url or None)
    results = [
        analyze_market(
            client=client,
            model=args.model,
            market=market,
            query=args.query,
            min_edge=args.min_edge,
            max_size=args.max_size,
            news_limit=args.news_limit,
            article_max_chars=args.article_max_chars,
        )
        for market in candidates
    ]

    best_result = max(results, key=lambda item: item["recommendation"]["edge"])

    if args.json:
        print(
            json.dumps(
                {"query": args.query, "best": best_result, "all": results}, indent=2
            )
        )
    else:
        print_text_report(best_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
