import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

def test_reduce_dimensions_tool_known_answers_7_8_9():
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
    
    # Get reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # 1. Test PCA (Known Answer 7)
    res_pca = reduce_dimensions_tool.invoke({"method": "pca"})
    pca_coords = res_pca.get("coordinates", [])
    exp_var = res_pca.get("explained_variance_ratio", [])
    
    assert len(pca_coords) == 8
    assert len(pca_coords[0]) == 2
    assert pytest.approx(pca_coords[0][0], abs=1e-2) == 2.3669
    assert pytest.approx(pca_coords[0][1], abs=1e-2) == 0.8871
    assert pytest.approx(exp_var[0], abs=1e-2) == 0.7532
    assert pytest.approx(exp_var[1], abs=1e-2) == 0.1965
    assert session.pending_figure is not None
    
    # 2. Test t-SNE (Known Answer 8: perplexity=2, random_state=42)
    res_tsne = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    tsne_coords = res_tsne.get("coordinates", [])
    assert len(tsne_coords) == 8
    assert len(tsne_coords[0]) == 2
    assert res_tsne.get("method") == "tsne"
    assert session.pending_figure is not None
    
    # 3. Test UMAP (Known Answer 9: n_neighbors=3, random_state=42)
    res_umap = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    umap_coords = res_umap.get("coordinates", [])
    assert len(umap_coords) == 8
    assert len(umap_coords[0]) == 2
    assert res_umap.get("method") == "umap"
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    """Test binarize_labels_tool on sample_fixture.csv (catA and catB)."""
    # Test error handling when no dataframe loaded
    empty_session = SessionState()
    empty_tools = make_reduction_tools(empty_session)
    binarize_tool_empty = next(t for t in empty_tools if t.name == "binarize_labels_tool")
    err_res = binarize_tool_empty.invoke({})
    assert "error" in err_res

    # Test with loaded sample fixture
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res["n_documents"] == 8
    assert res["n_classes"] == 2
    assert res["classes"] == ["catA", "catB"]

    one_hot = res["one_hot_matrix"]
    assert len(one_hot) == 8
    assert len(one_hot[0]) == 2

    # Rows 0-2 and 7 are catA -> [1, 0]
    assert one_hot[0] == [1, 0]
    # Rows 3-6 are catB -> [0, 1]
    assert one_hot[3] == [0, 1]

    assert "binarized_labels" in session.artifacts
