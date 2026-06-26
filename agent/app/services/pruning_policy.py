# app/services/pruning_policy.py
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.core.settings import settings

def run_pruning_policy():
    """Sweeps across all isolated RAG database partitions to evaluate storage health parameters."""
    client = QdrantClient(host=settings.qdrant.server_host, port=settings.qdrant.server_port)
    
    # 🚀 Track metrics across all distinct collection partitions deployed in production
    target_collections = ["general_memory", "research_papers", "vision_detection"]
    
    for col_name in target_collections:
        try:
            if not client.collection_exists(col_name):
                print(f"[{col_name.upper()}] Vector repository empty. Skipping sweep.")
                continue

            results = client.scroll(
                collection_name=col_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            points = results[0]
            if not points:
                print(f"[{col_name.upper()}] Vector repository empty. Skipping sweep.")
                continue

            current_timestamp = int(time.time())
            ids_slated_for_deletion = []

            for point in points:
                fact_node_id = point.id
                metadata_map = point.payload
                
                last_accessed_timestamp = metadata_map.get('last_accessed', current_timestamp)
                importance_coefficient = metadata_map.get('score_importance', 0.5)
                
                inactive_duration_seconds = current_timestamp - last_accessed_timestamp
                
                # Policy: Purge facts idle for over 24 hours with low importance scores (< 0.4)
                if inactive_duration_seconds > 86400 and importance_coefficient < 0.4:
                    ids_slated_for_deletion.append(fact_node_id)

            if ids_slated_for_deletion:
                client.delete(collection_name=col_name, points_selector=ids_slated_for_deletion)
                print(f"==> [Pruning Job] Successfully cleaned {len(ids_slated_for_deletion)} nodes inside [{col_name.upper()}].")
            else:
                print(f"==> [Pruning Job] [{col_name.upper()}] conforms to storage parameters. Purge skipped.")
        except Exception as col_err:
            print(f"==> [Pruning Job Warning] Active registry slot [{col_name.upper()}] error: {col_err}")

if __name__ == "__main__":
    run_pruning_policy()