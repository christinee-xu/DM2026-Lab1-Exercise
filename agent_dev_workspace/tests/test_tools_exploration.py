import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools


@pytest.fixture
def loaded_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student's build_dtm_tool
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def test_cosine_similarity_tool(loaded_session):
    """Known Answer #16: Check cosine similarities on sample_fixture.csv DTM."""
    tools = make_tools(loaded_session)
    cosine_sim_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Doc 0 vs 1 (both catA): ~0.9847
    res01 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 1})
    assert res01["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 3 (catA vs catB): ~0.0909
    res03 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 3})
    assert res03["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Doc 3 vs 4 (both catB): ~0.9847
    res34 = cosine_sim_tool.invoke({"doc_index1": 3, "doc_index2": 4})
    assert res34["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 7 (empty text doc): 0.0
    res07 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 7})
    assert res07["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)


def test_feature_correlation_matrix_tool(loaded_session):
    """Known Answer #19: Check top 20 variance feature correlation matrix on sample_fixture.csv."""
    tools = make_tools(loaded_session)
    feat_corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    res = feat_corr_tool.invoke({"top_n": 20})

    assert "terms" in res
    assert "correlation_matrix" in res

    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    corr_mat = np.array(res["correlation_matrix"])

    # Diagonal elements should all be 1.0
    assert np.allclose(np.diag(corr_mat), 1.0, atol=1e-3)

    # Key pair assertions based on Known Answer #19
    # alpha (index 0) vs beta (index 3) -> 0.8885
    assert corr_mat[0, 3] == pytest.approx(0.8885, abs=1e-3)

    # gamma (index 1) vs delta (index 2) -> 0.8307
    assert corr_mat[1, 2] == pytest.approx(0.8307, abs=1e-3)

    # alpha (index 0) vs gamma (index 1) -> -0.5718
    assert corr_mat[0, 1] == pytest.approx(-0.5718, abs=1e-3)

    # Check that pending figure was set
    assert loaded_session.pending_figure is not None
