"""Unit tests for src/sentiment/catalysts.py — pure functions, no network."""
from src.sentiment.catalysts import classify_events, has_risk


class TestClassifyEvents:
    def test_empty_titles(self):
        result = classify_events([])
        assert result["catalysts"] == []
        assert result["risks"] == []

    def test_none_titles(self):
        result = classify_events(None)
        assert result["catalysts"] == []

    def test_earnings_beat_is_catalyst(self):
        titles = ["NVDA beats earnings estimates by wide margin"]
        result = classify_events(titles)
        tags = [c["tag"] for c in result["catalysts"]]
        assert "財報優於預期" in tags

    def test_guidance_cut_is_risk(self):
        titles = ["Company lowers guidance for next quarter"]
        result = classify_events(titles)
        tags = [r["tag"] for r in result["risks"]]
        assert "財測下修" in tags

    def test_buyback_is_catalyst(self):
        titles = ["Apple announces $90 billion share repurchase program"]
        result = classify_events(titles)
        tags = [c["tag"] for c in result["catalysts"]]
        assert "庫藏股/回購" in tags

    def test_layoff_is_risk(self):
        titles = ["Tech giant announces 10% layoff across all divisions"]
        result = classify_events(titles)
        tags = [r["tag"] for r in result["risks"]]
        assert "裁員/重組" in tags

    def test_chinese_catalyst(self):
        titles = ["台積電獲單創單季新高，法人大幅調升財測"]
        result = classify_events(titles)
        tags = [c["tag"] for c in result["catalysts"]]
        assert "調升財測/上修" in tags or "訂單/接單" in tags

    def test_chinese_risk(self):
        titles = ["2330 外資賣超創近三年最大，主力大量解禁"]
        result = classify_events(titles)
        tags = [r["tag"] for r in result["risks"]]
        assert "解禁/大股東減持" in tags

    def test_no_duplicate_tags(self):
        titles = [
            "Company beats earnings expectations strongly",
            "Firm exceeds estimates by record margin",
        ]
        result = classify_events(titles)
        tags = [c["tag"] for c in result["catalysts"]]
        # 同一 tag 不應重複出現
        assert len(tags) == len(set(tags))

    def test_both_catalyst_and_risk(self):
        titles = [
            "Stock hits record high after earnings beat",
            "SEC investigates company for accounting irregularities",
        ]
        result = classify_events(titles)
        assert len(result["catalysts"]) > 0
        assert len(result["risks"]) > 0

    def test_neutral_headline_no_match(self):
        titles = ["Stock market opens flat on Tuesday"]
        result = classify_events(titles)
        assert result["catalysts"] == []
        assert result["risks"] == []


class TestHasRisk:
    def test_with_risks(self):
        ev = {"catalysts": [], "risks": [{"tag": "裁員", "headline": "..."}]}
        assert has_risk(ev) is True

    def test_without_risks(self):
        ev = {"catalysts": [{"tag": "財報優於預期", "headline": "..."}], "risks": []}
        assert has_risk(ev) is False

    def test_empty_dict(self):
        assert has_risk({}) is False

    def test_none(self):
        assert has_risk(None) is False
