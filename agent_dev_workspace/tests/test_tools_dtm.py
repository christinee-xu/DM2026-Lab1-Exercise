import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_known_answer_4():
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
    
    # Get build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    # Invoke build_dtm_tool
    res = build_dtm_tool.invoke({})
    
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert res.get("vocabulary") == expected_vocab, f"Expected vocab {expected_vocab}, got {res.get('vocabulary')}"
    assert res.get("n_documents") == 8
    assert res.get("n_terms") == 5
    assert res.get("non_zero") == 21
    assert res.get("total_elements") == 40
    assert pytest.approx(res.get("sparsity_pct"), abs=1e-3) == 47.5
    
    # Verify artifact storage
    assert "count_vectorizer" in session.artifacts, "build_dtm_tool must store count_vectorizer in session.artifacts"


def test_term_frequency_tool_known_answer_14():
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
    
    # Get tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    
    # Build DTM first
    build_dtm_tool.invoke({})
    
    # Invoke term_frequency_tool
    res = term_frequency_tool.invoke({})
    freqs = res.get("term_frequencies", {})
    
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    
    for term, expected_count in expected_freqs.items():
        assert freqs.get(term) == expected_count, f"Expected {term}={expected_count}, got {freqs.get(term)}"


def test_dtm_heatmap_tool_known_answer_21():
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
    
    # Get tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")
    
    # Build DTM first
    build_dtm_tool.invoke({})
    
    # Invoke dtm_heatmap_tool for top-left 3x2 corner
    res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    
    expected_slice = [[3, 1], [2, 1], [1, 1]]
    assert res.get("matrix_slice") == expected_slice, f"Expected slice {expected_slice}, got {res.get('matrix_slice')}"
    assert res.get("terms_included") == ["alpha", "always"], f"Expected terms ['alpha', 'always'], got {res.get('terms_included')}"
    
    # Verify plot output
    assert session.pending_figure is not None, "dtm_heatmap_tool must set session.pending_figure"
