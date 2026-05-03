import unittest

from agents.custom.edge_agent import (
    MarketContextLayer,
    build_analysis_prompt,
    build_baseline_analysis_prompt,
    extract_resolution_criteria,
    make_recommendation,
    to_market_snapshot,
)


class EdgeAgentTests(unittest.TestCase):
    def test_market_snapshot_parses_stringified_yes_no_market(self):
        snapshot = to_market_snapshot(
            {
                "id": "123",
                "question": "Will X happen?",
                "description": "A binary market.",
                "slug": "will-x-happen",
                "endDate": "2026-12-31T00:00:00Z",
                "liquidity": "1000",
                "volume": "5000",
                "volume24hr": "100",
                "spread": "0.01",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
                "restricted": True,
                "events": [{"slug": "will-x-happen-event"}],
            }
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.yes_price, 0.4)
        self.assertEqual(snapshot.no_price, 0.6)
        self.assertEqual(snapshot.url, "https://polymarket.com/event/will-x-happen-event")

    def test_recommendation_buys_yes_when_model_has_edge(self):
        snapshot = to_market_snapshot(
            {
                "id": "123",
                "question": "Will X happen?",
                "description": "A binary market.",
                "slug": "will-x-happen",
                "endDate": "2026-12-31T00:00:00Z",
                "liquidity": "1000",
                "volume": "5000",
                "volume24hr": "100",
                "spread": "0.01",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
            }
        )

        recommendation = make_recommendation(
            market=snapshot,
            analysis={"probability_yes": 0.58, "confidence": 0.8, "rationale": "Test edge."},
            min_edge=0.05,
            max_size=0.10,
        )

        self.assertEqual(recommendation.action, "BUY_YES")
        self.assertGreater(recommendation.size_fraction, 0.0)

    def test_recommendation_passes_when_edge_is_too_small(self):
        snapshot = to_market_snapshot(
            {
                "id": "123",
                "question": "Will X happen?",
                "description": "A binary market.",
                "slug": "will-x-happen",
                "endDate": "2026-12-31T00:00:00Z",
                "liquidity": "1000",
                "volume": "5000",
                "volume24hr": "100",
                "spread": "0.01",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
            }
        )

        recommendation = make_recommendation(
            market=snapshot,
            analysis={"probability_yes": 0.43, "confidence": 0.9, "rationale": "No real edge."},
            min_edge=0.05,
            max_size=0.10,
        )

        self.assertEqual(recommendation.action, "PASS")
        self.assertEqual(recommendation.size_fraction, 0.0)

    def test_extract_resolution_criteria_prefers_resolution_language(self):
        description = (
            "This market asks about an event.\n\n"
            'This market will resolve to "Yes" if the event happens before June 1, 2026. '
            'Otherwise, this market will resolve to "No".\n\n'
            "The primary resolution source will be official government reporting."
        )

        resolution_criteria = extract_resolution_criteria(description)

        self.assertIn('resolve to "Yes"', resolution_criteria)
        self.assertIn("resolution source", resolution_criteria.lower())

    def test_build_analysis_prompt_embeds_context_layer(self):
        snapshot = to_market_snapshot(
            {
                "id": "123",
                "question": "Will X happen?",
                "description": "This market will resolve to Yes if X happens.",
                "slug": "will-x-happen",
                "endDate": "2026-12-31T00:00:00Z",
                "liquidity": "1000",
                "volume": "5000",
                "volume24hr": "100",
                "spread": "0.01",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
            }
        )

        prompt = build_analysis_prompt(
            "x event",
            snapshot,
            MarketContextLayer(
                resolution_criteria="Resolve by official filing.",
                news_articles=[],
            ),
        )

        self.assertIn("Context layer", prompt)
        self.assertIn("Resolve by official filing.", prompt)

    def test_build_baseline_analysis_prompt_omits_context_layer(self):
        snapshot = to_market_snapshot(
            {
                "id": "123",
                "question": "Will X happen?",
                "description": "This market will resolve to Yes if X happens.",
                "slug": "will-x-happen",
                "endDate": "2026-12-31T00:00:00Z",
                "liquidity": "1000",
                "volume": "5000",
                "volume24hr": "100",
                "spread": "0.01",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
            }
        )

        prompt = build_baseline_analysis_prompt("x event", snapshot)

        self.assertNotIn("Context layer", prompt)
        self.assertIn("Question: Will X happen?", prompt)


if __name__ == "__main__":
    unittest.main()
