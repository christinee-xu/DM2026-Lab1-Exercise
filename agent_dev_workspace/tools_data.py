import os
import pathlib
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

# Ensure NLTK punkt data is downloaded
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)


def make_tools(session):
    @tool
    def tokenize_tool() -> dict:
        """Tokenizes each document's text column into lowercased unigrams.

        Returns:
            dict: Summary containing result_id, n_documents, total_unigrams, sample_tokens, and tokens.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or text column available in session."}

        df = session.dataframe

        tokens_list = []
        for text in df["text"].fillna(""):
            if text:
                tokens = nltk.word_tokenize(str(text).lower())
            else:
                tokens = []
            tokens_list.append(tokens)

        df["unigrams"] = tokens_list
        session.artifacts["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        total_unigrams = sum(len(t) for t in tokens_list)
        sample_tokens = tokens_list[:3] if len(tokens_list) >= 3 else tokens_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "total_unigrams": total_unigrams,
            "sample_tokens": sample_tokens,
            "tokens": tokens_list,
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """Lists files and directories in the workspace or a specified subdirectory.

        Args:
            subdirectory (str): Optional subfolder path relative to workspace root (e.g., "newdataset").

        Returns:
            dict: Summary containing result_id, subdirectory, directories, and files.
        """
        base_dir = pathlib.Path(".")
        target_dir = base_dir / subdirectory if subdirectory else base_dir

        if not target_dir.exists() or not target_dir.is_dir():
            return {"error": f"Directory '{subdirectory}' does not exist or is not a directory."}

        dirs = []
        files = []
        for item in target_dir.iterdir():
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            if item.is_dir():
                dirs.append(item.name)
            elif item.is_file():
                files.append(item.name)

        dirs.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": dirs,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5, mode: str = "head") -> dict:
        """Inspects the working DataFrame rows and structure.

        Args:
            n (int, optional): Number of rows to inspect. Defaults to 5.
            mode (str, optional): Mode of inspection - "head", "tail", or "sample". Defaults to "head".

        Returns:
            dict: Summary containing result_id, shape, columns, mode, and sample_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe available in session. Load a dataset first."}

        df = session.dataframe
        n_rows, n_cols = df.shape
        k = min(n, n_rows)

        if mode == "tail":
            subset_df = df.tail(k)
        elif mode == "sample":
            subset_df = df.sample(n=k)
        else:
            subset_df = df.head(k)

        sample_rows = subset_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "shape": [n_rows, n_cols],
            "columns": list(df.columns),
            "mode": mode,
            "sample_rows": sample_rows,
        }

        session.store_result("inspect_data_tool", {"n": n, "mode": mode}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks for missing or empty text values in the working dataset.

        Returns:
            dict: Summary containing result_id, total_documents, missing_count, and missing_indices.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or text column available in session."}

        df = session.dataframe

        missing_indices = []
        for idx, text in enumerate(df["text"]):
            if text is None or (isinstance(text, str) and text.strip() == "") or (hasattr(text, "isna") and text.isna()):
                missing_indices.append(idx)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_documents": len(df),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for duplicate documents in the working dataset and optionally drops them.

        Args:
            drop (bool): If True, drops all copies of duplicate rows using keep=False.

        Returns:
            dict: Summary containing result_id, n_duplicates, dropped, and n_documents_remaining.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or text column available in session."}

        df = session.dataframe

        duplicated_mask = df.duplicated(subset=["text"], keep="first")
        n_duplicates = int(duplicated_mask.sum())

        if drop and n_duplicates > 0:
            df.drop_duplicates(subset=["text"], keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "n_duplicates": n_duplicates,
            "dropped": drop,
            "n_documents_remaining": len(session.dataframe),
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: int = None) -> dict:
        """Subsamples the working DataFrame randomly.

        Args:
            n (int): Number of documents to sample.
            random_state (int, optional): Random seed for reproducibility.

        Returns:
            dict: Summary containing result_id, sampled_indices, n_sampled, and sample_text.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or text column available in session."}

        df = session.dataframe
        sample_size = min(n, len(df))

        sampled_df = df.sample(n=sample_size, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()
        sample_text = sampled_df["text"].fillna("").tolist()

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sampled_indices": sampled_indices,
            "n_sampled": len(sampled_indices),
            "sample_text": sample_text,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Computes summary statistics on document text length and creates a box plot by category.

        Returns:
            dict: Summary containing result_id, overall_stats, per_category_stats, and text_lengths.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or text column available in session."}

        df = session.dataframe.copy()
        df["text_length"] = df["text"].fillna("").apply(len)

        overall_desc = df["text_length"].describe().to_dict()

        per_category_stats = {}
        category_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        if category_col:
            for cat, group in df.groupby(category_col):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (chars)")
            plt.tight_layout()

            session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_desc,
            "per_category_stats": per_category_stats,
            "text_lengths": df["text_length"].tolist(),
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
