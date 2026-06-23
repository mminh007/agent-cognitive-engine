# app/retrieval/rrf.py
from typing import List, Dict

class RFF:
     
    @staticmethod
    def _reciprocal_rank_fusion(dense_results: List[Dict], lexical_results: List[Dict], k: int = 60) -> List[Dict]:
            """
            Executes Reciprocal Rank Fusion (RRF) to score and re-rank documents 
            based on their ordinal positional indices retrieved from multi-modal pipelines.

            Args:
                dense_results (List[Dict]): Candidates extracted via dense semantic vector lookup.
                lexical_results (List[Dict]): Candidates extracted via keyword-matching BM25 calculations.
                k (int, optional): RRF constant parameter scale mitigating rank noise anomalies. Defaults to 60.

            Returns:
                List[Dict]: Consolidated corpus elements sorted descending by aggregated RRF scores.
            """
            rrf_scores = {}
            doc_mapping = {}

            # Process position ranks originating from Dense Semantic Retrieval
            for rank, doc in enumerate(dense_results):
                doc_id = doc["id"]
                doc_mapping[doc_id] = doc
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)

            # Process position ranks originating from Lexical Keyword Search (BM25)
            for rank, doc in enumerate(lexical_results):
                doc_id = doc["id"]
                doc_mapping[doc_id] = doc
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)

            # Re-rank elements by descending score
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            final_results = []
            for doc_id, score in sorted_docs:
                hydrated_doc = doc_mapping[doc_id].copy()
                hydrated_doc["rrf_score"] = score
                final_results.append(hydrated_doc)
                
            return final_results