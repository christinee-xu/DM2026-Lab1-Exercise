import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> dict:
        """Computes cosine similarity between two document vectors in session.feature_matrix.

        Args:
            doc_index1 (int): 0-based index of the first document.
            doc_index2 (int): 0-based index of the second document.

        Returns:
            dict: Summary containing result_id, doc_index1, doc_index2, cosine_similarity, text1, and text2.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix available in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        n_docs = X.shape[0]

        if doc_index1 < 0 or doc_index1 >= n_docs:
            return {"error": f"doc_index1 {doc_index1} is out of bounds for matrix with {n_docs} documents."}
        if doc_index2 < 0 or doc_index2 >= n_docs:
            return {"error": f"doc_index2 {doc_index2} is out of bounds for matrix with {n_docs} documents."}

        v1 = X[doc_index1]
        v2 = X[doc_index2]

        sim_val = float(cosine_similarity(v1, v2)[0][0])
        if np.isnan(sim_val):
            sim_val = 0.0

        text1 = ""
        text2 = ""
        if session.dataframe is not None and "text" in session.dataframe.columns:
            text1 = str(session.dataframe["text"].iloc[doc_index1])
            text2 = str(session.dataframe["text"].iloc[doc_index2])

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "cosine_similarity": round(sim_val, 4),
            "text1": text1,
            "text2": text2,
        }

        session.store_result("cosine_similarity_tool", {"doc_index1": doc_index1, "doc_index2": doc_index2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Computes feature-vs-feature correlation heatmap over top terms by variance.

        Args:
            top_n (int, optional): Number of highest variance features to include. Defaults to 20.

        Returns:
            dict: Summary containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM available in session. Run build_dtm_tool first."}

        X_dense = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)

        n_features = X_dense.shape[1]
        if n_features == 0:
            return {"error": "Feature matrix has 0 features."}

        variances = np.var(X_dense, axis=0)
        actual_top_n = min(top_n, n_features)

        # Get indices of top variance terms sorted descending
        top_indices = np.argsort(variances)[::-1][:actual_top_n]

        selected_X = X_dense[:, top_indices]
        selected_terms = [feature_names[i] for i in top_indices]

        # Pearson correlation matrix across selected features (columns)
        # rowvar=False means columns are variables
        corr_matrix = np.corrcoef(selected_X, rowvar=False)

        # Handle 1x1 matrix case or NaNs (e.g. constant terms)
        if actual_top_n == 1:
            corr_matrix = np.array([[1.0]])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            cmap="vlag",
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_top_n} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        corr_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        result_id = session.next_result_id("feat_corr")
        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
