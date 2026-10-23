import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags near-constant terms by variance for removal from session.feature_matrix.

        Args:
            threshold (float): Minimum variance threshold to keep a term.

        Returns:
            dict: Summary containing result_id, threshold, kept_terms, removed_terms, n_kept, n_removed, and term_variances.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available in session. Run build_dtm_tool first."}

        X_dense = session.feature_matrix.toarray()
        terms = session.feature_names

        variances = np.var(X_dense, axis=0)

        kept_terms = []
        removed_terms = []
        term_variances = {}

        report_data = []

        for term, var in zip(terms, variances):
            var_float = float(var)
            term_variances[term] = round(var_float, 4)
            report_data.append({"term": term, "variance": var_float})

            if var_float >= threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        full_report_df = pd.DataFrame(report_data).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "term_variances": term_variances,
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=full_report_df)
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filters terms by Pearson correlation with a binary target category vector.

        Args:
            target_class (str): Target category name to correlate against (e.g., 'catB').
            threshold (float): Minimum absolute Pearson correlation |r| to keep a term.

        Returns:
            dict: Summary containing result_id, target_class, threshold, kept_terms, removed_terms, and correlations.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available in session. Run build_dtm_tool first."}
        if session.labels is None:
            return {"error": "No category labels available in session."}

        X_dense = session.feature_matrix.toarray()
        terms = session.feature_names
        labels = session.labels

        y = np.array([1 if label == target_class else 0 for label in labels], dtype=float)

        correlations = {}
        kept_terms = []
        removed_terms = []
        report_data = []

        for i, term in enumerate(terms):
            x = X_dense[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(x, y)
                if np.isnan(r):
                    r = 0.0

            r_float = float(r)
            correlations[term] = round(r_float, 4)
            report_data.append({"term": term, "pearson_r": r_float})

            if abs(r_float) >= threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        full_report_df = pd.DataFrame(report_data).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "correlations": correlations,
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=full_report_df)
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filters terms by Spearman rank correlation with a binary target category vector.

        Args:
            target_class (str): Target category name to correlate against (e.g., 'catB').
            threshold (float): Minimum absolute Spearman correlation |rho| to keep a term.

        Returns:
            dict: Summary containing result_id, target_class, threshold, kept_terms, removed_terms, and correlations.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available in session. Run build_dtm_tool first."}
        if session.labels is None:
            return {"error": "No category labels available in session."}

        X_dense = session.feature_matrix.toarray()
        terms = session.feature_names
        labels = session.labels

        y = np.array([1 if label == target_class else 0 for label in labels], dtype=float)

        correlations = {}
        kept_terms = []
        removed_terms = []
        report_data = []

        for i, term in enumerate(terms):
            x = X_dense[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                rho = 0.0
            else:
                res = spearmanr(x, y)
                rho = res.statistic if hasattr(res, "statistic") else res[0]
                if np.isnan(rho):
                    rho = 0.0

            rho_float = float(rho)
            correlations[term] = round(rho_float, 4)
            report_data.append({"term": term, "spearman_r": rho_float})

            if abs(rho_float) >= threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        full_report_df = pd.DataFrame(report_data).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "correlations": correlations,
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=full_report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
