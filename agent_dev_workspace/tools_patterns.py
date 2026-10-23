import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 1,
    ) -> dict:
        """Mines frequent patterns for a specific category using PAMI algorithms.

        Args:
            category_name (str): Name of the category/class to mine.
            filtering_method (str): Filtering approach ('variance', 'tfidf', or 'term_frequency').
            algorithm (str): Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup (int): Minimum support count (for fpgrowth and maxfpgrowth).
            k (int): Number of top patterns to find (for topk algorithm).

        Returns:
            dict: Summary containing result_id, category_name, filtering_method, algorithm, and patterns.
        """
        if session.dataframe is None:
            return {"error": "No dataframe available in session."}

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        if not cat_col:
            return {"error": "No category column found in session.dataframe."}

        cat_df = df[df[cat_col] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Step 1: Category-local DTM with English stop words
        vec = CountVectorizer(stop_words="english")
        texts = cat_df["text"].fillna("")
        try:
            X = vec.fit_transform(texts)
        except ValueError:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }

        vocab = vec.get_feature_names_out().tolist()
        if not vocab:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "empty vocabulary",
            }

        X_dense = X.toarray()

        # Step 2 & 3: Apply vocabulary filtering
        if filtering_method == "variance":
            scores = np.var(X_dense, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "term_frequency":
            scores = X_dense.sum(axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            X_tfidf = tfidf_transformer.fit_transform(X).toarray()
            scores = X_tfidf.mean(axis=0)
            p20 = np.percentile(scores, 20)
            mask = scores >= p20
        else:
            return {"error": f"Unknown filtering_method '{filtering_method}'."}

        kept_vocab = [vocab[i] for i in range(len(vocab)) if mask[i]]
        if not kept_vocab:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category": category_name}, summary)
            return summary

        X_filtered = X_dense[:, mask]

        # Step 4: Convert to transactional DB and mine with PAMI
        transaction_df = pd.DataFrame(X_filtered, columns=kept_vocab)
        
        os.makedirs("patterns", exist_ok=True)
        db_file = f"patterns/db_{category_name}_{filtering_method}.txt"
        out_file = f"patterns/out_{category_name}_{filtering_method}_{algorithm}.txt"

        converter = DF2DB.DF2DB(transaction_df, "dense")
        converter.convert2TransactionalDatabase(db_file, ">=", 1)

        if algorithm == "fpgrowth":
            miner = FPGrowth.FPGrowth(db_file, min_sup)
        elif algorithm == "topk":
            miner = FAE.FAE(db_file, k)
        elif algorithm == "maxfpgrowth":
            miner = MaxFPGrowth.MaxFPGrowth(db_file, min_sup)
        else:
            return {"error": f"Unknown algorithm '{algorithm}'."}

        miner.startMine()
        try:
            miner.save(out_file)
        except Exception:
            pass

        # Step 5: Read patterns output
        patterns = []
        p_df = None
        if hasattr(miner, "getPatternsAsDataFrame"):
            try:
                p_df = miner.getPatternsAsDataFrame()
            except Exception:
                p_df = None

        if p_df is not None and not p_df.empty:
            for _, row in p_df.iterrows():
                pat_str = str(row.iloc[0])
                sup_val = int(row.iloc[1])
                patterns.append({"pattern": pat_str, "support": sup_val})
        elif os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            with open(out_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
                        if len(parts) >= 2:
                            patterns.append({"pattern": parts[0].strip(), "support": int(float(parts[1]))})

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns,
            "n_patterns": len(patterns),
        }

        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
        return summary

    return [mine_patterns_tool]
