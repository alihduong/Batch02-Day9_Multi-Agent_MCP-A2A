"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.query import MetadataQuery

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "DrugLawDocs"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = _get_model()
    query_embedding = model.encode(query).tolist()

    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        output = [
            {
                "content": obj.properties["content"],
                "score": 1 - obj.metadata.distance,
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "type": obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                },
            }
            for obj in results.objects
        ]
    finally:
        client.close()

    return sorted(output, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
