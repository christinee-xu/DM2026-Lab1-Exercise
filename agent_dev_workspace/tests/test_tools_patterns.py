import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_tool_known_answers_17_and_20():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Get mine_patterns_tool
    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    
    # Test 1: catA with tfidf + fpgrowth (min_sup=1) -> alpha, support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    pats_A = res_catA.get("patterns", [])
    assert len(pats_A) == 1
    assert "alpha" in pats_A[0]["pattern"]
    assert pats_A[0]["support"] == 3

    # Test 2: catB with tfidf + fpgrowth (min_sup=1) -> gamma, support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    pats_B = res_catB.get("patterns", [])
    assert len(pats_B) == 1
    assert "gamma" in pats_B[0]["pattern"]
    assert pats_B[0]["support"] == 4

    # Test 3: Zero-terms survived case (variance filtering on this fixture)
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth"
    })
    assert res_var.get("patterns") == []
    assert "note" in res_var or res_var.get("n_patterns") == 0

    # Test 4: maxfpgrowth known answer 20 (returns 0 patterns on single term DB)
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max.get("patterns") == []
