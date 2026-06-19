# app/services/pruning_policy.py
import time
import chromadb
from app.core.settings import settings

def run_pruning_policy():
    """Sweeps across all isolated RAG database partitions to evaluate storage health parameters."""
    client = chromadb.PersistentClient(path=settings.chroma.path)
    
    # 🚀 Track metrics across all distinct collection partitions deployed in production
    target_collections = ["general_memory", "research_papers", "vision_detection"]
    
    for col_name in target_collections:
        try:
            collection = client.get_collection(name=col_name)
            results = collection.get()
            if not results or not results['ids']:
                print(f"[{col_name.upper()}] Vector repository empty. Skipping sweep.")
                continue

            current_timestamp = int(time.time())
            ids_slated_for_deletion = []

            for i in range(len(results['ids'])):
                fact_node_id = results['ids'][i]
                metadata_map = results['metadatas'][i]
                
                last_accessed_timestamp = metadata_map.get('last_accessed', current_timestamp)
                importance_coefficient = metadata_map.get('score_importance', 0.5)
                
                inactive_duration_seconds = current_timestamp - last_accessed_timestamp
                
                # Policy: Purge facts idle for over 24 hours with low importance scores (< 0.4)
                if inactive_duration_seconds > 86400 and importance_coefficient < 0.4:
                    ids_slated_for_deletion.append(fact_node_id)

            if ids_slated_for_deletion:
                collection.delete(ids=ids_slated_for_deletion)
                print(f"==> [Pruning Job] Successfully cleaned {len(ids_slated_for_deletion)} nodes inside [{col_name.upper()}].")
            else:
                print(f"==> [Pruning Job] [{col_name.upper()}] conforms to storage parameters. Purge skipped.")
        except Exception as col_err:
            print(f"==> [Pruning Job Warning] Active registry slot [{col_name.upper()}] not yet initialized on disk.")

if __name__ == "__main__":
    run_pruning_policy()