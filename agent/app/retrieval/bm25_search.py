from rank_bm25 import BM25Okapi
from typing import List, Dict

class BM25Retriever:
    """Xử lý luồng Sparse Lexical Search độc lập."""
    
    def retrieve(self, query: str, corpus_docs: List[Dict], limit: int) -> List[Dict]:
        if not corpus_docs:
            return []

        # Tokenize queries và documents
        tokenized_query = query.lower().split(" ")
        tokenized_corpus = [doc["text"].lower().split(" ") for doc in corpus_docs]
        
        bm25_engine = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25_engine.get_scores(tokenized_query)
        
        lexical_results = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:  # Chỉ giữ lại các doc có độ trùng khớp từ khóa thực sự
                doc_copy = corpus_docs[idx].copy()
                doc_copy["bm25_score"] = score
                lexical_results.append(doc_copy)
        
        # Sắp xếp và cắt lấy số lượng ứng viên tốt nhất (limit * 2 để lát nữa RRF mix lại)
        lexical_results = sorted(lexical_results, key=lambda x: x["bm25_score"], reverse=True)[:limit * 2]
        
        return lexical_results