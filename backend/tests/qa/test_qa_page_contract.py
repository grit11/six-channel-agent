"""知识页 QA-only 结构契约测试。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "app" / "static" / "dashboard.html"
JS = ROOT / "app" / "static" / "knowledge_qa.js"


def test_knowledge_view_contract() -> None:
    """知识页保留条文速查 ID，并使用独立 qa2 命名空间脚本。"""
    source = HTML.read_text(encoding="utf-8")
    script = JS.read_text(encoding="utf-8")
    assert 'data-view="knowledge"' in source
    for item in ("knowledgeQuery", "knowledgeSearch", "knowledgeStatus", "knowledgeResults"):
        assert f'id="{item}"' in source
    assert "qa2-layout" in source and "knowledge_qa.js" in source
    assert "textContent" in script and "innerHTML" not in script
    assert "qa2SourceToggle" in source and "qa2-open" in script


def test_other_views_have_no_qa2_markup() -> None:
    """工作台、轨迹、历史页面不包含 QA 专属 DOM。"""
    source = HTML.read_text(encoding="utf-8")
    for view in ("workbench", "trace", "history"):
        start = source.index(f'data-view="{view}"')
        end = source.find('<section class="view"', start + 10)
        fragment = source[start:] if end < 0 else source[start:end]
        assert "qa2-" not in fragment
