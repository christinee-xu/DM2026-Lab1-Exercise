import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_tokenize_tool_known_answer_10():
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
    
    # Get tokenize_tool
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    
    # Invoke tokenize_tool
    res = tokenize_tool.invoke({})
    
    # Known answer 10 expected tokens (NLTK word_tokenize on lowercased text)
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    # Extract tokens from returned dict or session state
    tokens = res.get("tokens") if isinstance(res, dict) else None
    if tokens is None and "tokens" in session.artifacts:
        tokens = session.artifacts["tokens"]
        
    assert tokens is not None, "tokenize_tool did not return or store tokens list"
    assert len(tokens) == 8, f"Expected 8 tokenized documents, got {len(tokens)}"
    
    for i, expected in enumerate(expected_tokens):
        assert tokens[i] == expected, f"Mismatch at document {i}: expected {expected}, got {tokens[i]}"


def test_list_files_tool_known_answer_11():
    session = SessionState()
    
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")
    
    # Invoke list_files_tool for subdirectory="newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    assert res.get("directories") == [], f"Expected empty directories list, got {res.get('directories')}"
    assert res.get("files") == ["Reddit-stock-sentiment.csv"], f"Expected ['Reddit-stock-sentiment.csv'], got {res.get('files')}"


def test_inspect_data_tool():
    """Custom test for inspect_data_tool covering head, tail, sample, and error states."""
    # Test error handling when no dataframe loaded
    empty_session = SessionState()
    empty_tools = make_tools(empty_session)
    inspect_tool_empty = next(t for t in empty_tools if t.name == "inspect_data_tool")
    err_res = inspect_tool_empty.invoke({})
    assert "error" in err_res

    # Test with loaded dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Mode: head (default)
    res_head = inspect_tool.invoke({"n": 3, "mode": "head"})
    assert res_head["shape"] == [8, 3]
    assert "text" in res_head["columns"]
    assert len(res_head["sample_rows"]) == 3
    assert res_head["sample_rows"][0]["text"] == "always alpha alpha alpha beta"

    # Mode: tail (checking category_name column)
    res_tail = inspect_tool.invoke({"n": 2, "mode": "tail"})
    assert len(res_tail["sample_rows"]) == 2
    assert res_tail["sample_rows"][-1]["category_name"] == "catA"

    # Mode: sample
    res_sample = inspect_tool.invoke({"n": 4, "mode": "sample"})
    assert res_sample["mode"] == "sample"
    assert len(res_sample["sample_rows"]) == 4
    for row in res_sample["sample_rows"]:
        assert "text" in row and "category_name" in row


def test_check_missing_tool_known_answer_2():
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
    
    # Get check_missing_tool
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")
    
    # Invoke check_missing_tool
    res = check_missing_tool.invoke({})
    
    assert res.get("missing_count") == 1, f"Expected 1 missing row, got {res.get('missing_count')}"
    assert res.get("missing_indices") == [7], f"Expected missing index [7], got {res.get('missing_indices')}"


def test_check_duplicates_tool_known_answer_3():
    # Fresh session for check without dropping
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")
    
    res1 = check_duplicates_tool1.invoke({"drop": False})
    assert res1.get("n_duplicates") == 1, f"Expected 1 duplicate, got {res1.get('n_duplicates')}"
    assert res1.get("n_documents_remaining") == 8, f"Expected 8 documents remaining when drop=False, got {res1.get('n_documents_remaining')}"

    # Fresh session for check WITH dropping (keep=False leaves 6 rows)
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")
    
    res2 = check_duplicates_tool2.invoke({"drop": True})
    assert res2.get("n_duplicates") == 1, f"Expected 1 duplicate, got {res2.get('n_duplicates')}"
    assert res2.get("dropped") is True, "Expected dropped=True"
    assert res2.get("n_documents_remaining") == 6, f"Expected 6 documents remaining when drop=True (keep=False), got {res2.get('n_documents_remaining')}"


def test_sample_data_tool_known_answer_13():
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
    
    # Get sample_data_tool
    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")
    
    # Invoke sample_data_tool with n=4, random_state=42
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    
    assert res.get("n_sampled") == 4, f"Expected 4 sampled rows, got {res.get('n_sampled')}"
    assert res.get("sampled_indices") == [1, 5, 0, 7], f"Expected indices [1, 5, 0, 7], got {res.get('sampled_indices')}"


def test_describe_data_tool_known_answer_18():
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
    
    # Get describe_data_tool
    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")
    
    # Invoke describe_data_tool
    res = describe_data_tool.invoke({})
    
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert res.get("text_lengths") == expected_lengths, f"Expected {expected_lengths}, got {res.get('text_lengths')}"
    
    overall = res.get("overall_stats", {})
    assert pytest.approx(overall.get("mean"), abs=1e-3) == 19.875
    assert pytest.approx(overall.get("count"), abs=1e-3) == 8.0
    
    per_cat = res.get("per_category_stats", {})
    assert "catA" in per_cat and "catB" in per_cat
    assert pytest.approx(per_cat["catA"]["mean"], abs=1e-3) == 17.25
    assert pytest.approx(per_cat["catB"]["mean"], abs=1e-3) == 22.5
    
    # Verify plotting requirement
    assert session.pending_figure is not None, "describe_data_tool must set session.pending_figure"
