import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import LabelBinarizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: int = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Reduces document-term matrix dimensions to 2D using PCA, t-SNE, or UMAP and plots a scatter plot.

        Args:
            method (str): Reduction technique ('pca', 'tsne', or 'umap').
            n_components (int): Target number of dimensions (default 2).
            random_state (int, optional): Random seed for reproducibility.
            perplexity (float): Perplexity parameter for t-SNE (used for small datasets).
            n_neighbors (int): Number of neighbors parameter for UMAP.

        Returns:
            dict: Summary containing result_id, method, coordinates, and explained_variance_ratio.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix available in session. Run build_dtm_tool first."}

        X_dense = session.feature_matrix.toarray()
        n_samples = X_dense.shape[0]

        explained_variance_ratio = None

        if method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        elif method.lower() == "tsne":
            # Pass integer perplexity to TSNE
            p_val = int(perplexity) if perplexity < n_samples else max(1, n_samples - 1)
            reducer = TSNE(
                n_components=n_components,
                perplexity=p_val,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)
        elif method.lower() == "umap":
            eff_neighbors = min(int(n_neighbors), max(2, n_samples - 1))
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=eff_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)
        else:
            return {"error": f"Unknown reduction method '{method}'."}

        coords_rounded = [[round(float(c[0]), 4), round(float(c[1]), 4)] for c in coords]
        session.artifacts["reduced_coordinates"] = coords_rounded

        labels = session.labels if session.labels is not None else [f"Doc {i}" for i in range(n_samples)]

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels,
            palette="Set2",
            s=100,
            ax=ax,
        )
        ax.set_title(f"2D Dimension Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "coordinates": coords_rounded,
            "explained_variance_ratio": explained_variance_ratio,
        }

        session.store_result("reduce_dimensions_tool", {"method": method}, summary)
        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes the category labels of the dataset into a binary matrix.

        Returns:
            dict: Summary containing result_id, classes, n_documents, n_classes, and one_hot_matrix.
        """
        if session.dataframe is None:
            return {"error": "No dataframe available in session. Load a dataset first."}

        category_col = "category_name" if "category_name" in session.dataframe.columns else ("category" if "category" in session.dataframe.columns else None)
        if not category_col:
            return {"error": "No category column found in session.dataframe."}

        labels = session.dataframe[category_col].tolist()

        lb = LabelBinarizer()
        binarized = lb.fit_transform(labels)

        classes = [str(c) for c in lb.classes_]
        if len(classes) == 2 and binarized.shape[1] == 1:
            binarized = np.hstack((1 - binarized, binarized))

        binarized_list = binarized.tolist()
        session.artifacts["binarized_labels"] = binarized_list

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "n_documents": len(labels),
            "n_classes": len(classes),
            "one_hot_matrix": binarized_list,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
