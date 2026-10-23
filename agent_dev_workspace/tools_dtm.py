from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
    ) -> dict:
        """Builds a document-term matrix from session.dataframe using CountVectorizer.

        Args:
            ngram_range (List[int]): Lower and upper boundary of n-gram values (e.g. [1, 1]).
            max_df (Union[float, int]): Maximum document frequency threshold.
            min_df (Union[float, int]): Minimum document frequency threshold.

        Returns:
            dict: Summary containing result_id, n_documents, n_terms, vocabulary, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or text column available in session."}

        df = session.dataframe

        ng_tuple = tuple(ngram_range) if isinstance(ngram_range, list) else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ng_tuple,
            max_df=max_df,
            min_df=min_df,
        )

        texts = df["text"].fillna("")
        X = vectorizer.fit_transform(texts)

        session.feature_matrix = X
        session.feature_names = vectorizer.get_feature_names_out().tolist()
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in df.columns:
            session.set_labels(df["category_name"].values)
        elif "category" in df.columns:
            session.set_labels(df["category"].values)

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - (non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "vocabulary": session.feature_names,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "max_df": max_df, "min_df": min_df}, summary)
        return summary

    @tool
    def term_frequency_tool() -> dict:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Returns:
            dict: Summary containing result_id, term_frequencies, and top_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available in session. Run build_dtm_tool first."}

        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).ravel()

        term_freqs = {
            term: int(count)
            for term, count in zip(session.feature_names, col_sums)
        }

        sorted_freqs = dict(sorted(term_freqs.items(), key=lambda item: item[1], reverse=True))
        top_terms = [[term, count] for term, count in sorted_freqs.items()]

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "term_frequencies": sorted_freqs,
            "top_terms": top_terms,
        }

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Visualizes a positional slice of the document-term matrix as a heatmap.

        Args:
            n_terms (int): Maximum number of terms (columns) to include from index 0.
            n_documents (int): Maximum number of documents (rows) to include from index 0.

        Returns:
            dict: Summary containing result_id, matrix_slice, terms_included, n_documents_included, and n_terms_included.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        vocab = session.feature_names

        n_docs_slice = min(n_documents, X.shape[0])
        n_terms_slice = min(n_terms, X.shape[1])

        X_slice = X[:n_docs_slice, :n_terms_slice].toarray()
        terms_included = vocab[:n_terms_slice]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            X_slice,
            annot=True,
            fmt="d",
            xticklabels=terms_included,
            yticklabels=[f"Doc {i}" for i in range(n_docs_slice)],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap ({n_docs_slice} Docs x {n_terms_slice} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "matrix_slice": X_slice.tolist(),
            "terms_included": terms_included,
            "n_documents_included": n_docs_slice,
            "n_terms_included": n_terms_slice,
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
