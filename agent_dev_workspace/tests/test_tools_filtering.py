import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool_known_answer_5():
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
    
    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Get variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    # Invoke variance_filter_tool with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})
    
    kept = res.get("kept_terms", [])
    removed = res.get("removed_terms", [])
    variances = res.get("term_variances", {})
    
    # Assert variances match known answer 5
    assert pytest.approx(variances.get("delta"), abs=1e-3) == 0.25
    assert pytest.approx(variances.get("always"), abs=1e-3) == 0.109375
    
    # Assert filtering decisions at threshold 0.15
    assert "always" in removed, "always (var=0.109375) should be removed at threshold 0.15"
    assert "delta" in kept, "delta (var=0.25) should be kept at threshold 0.15"
    assert set(kept) == {"alpha", "beta", "delta", "gamma"}
    assert removed == ["always"]


def test_pearson_filter_tool_known_answer_6():
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
    
    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Get pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    # Invoke pearson_filter_tool with target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB"})
    corrs = res.get("correlations", {})
    
    assert pytest.approx(corrs.get("alpha"), abs=1e-3) == -0.6882
    assert pytest.approx(corrs.get("always"), abs=1e-3) == 0.3780
    assert pytest.approx(corrs.get("beta"), abs=1e-3) == -0.7746
    assert pytest.approx(corrs.get("delta"), abs=1e-3) == 1.0000
    assert pytest.approx(corrs.get("gamma"), abs=1e-3) == 0.8307


def test_spearman_filter_tool_known_answer_6():
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
    
    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Get spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    
    # Invoke spearman_filter_tool with target_class="catB"
    res = spearman_filter_tool.invoke({"target_class": "catB"})
    corrs = res.get("correlations", {})
    
    assert pytest.approx(corrs.get("alpha"), abs=1e-3) == -0.7500
    assert pytest.approx(corrs.get("always"), abs=1e-3) == 0.3780
    assert pytest.approx(corrs.get("beta"), abs=1e-3) == -0.7746
    assert pytest.approx(corrs.get("delta"), abs=1e-3) == 1.0000
    assert pytest.approx(corrs.get("gamma"), abs=1e-3) == 0.9363
