from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    """
    Load the embedding model only once.
    """

    global _model

    if _model is None:

        print("\nLoading Embedding Model...")

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print("Embedding Model Loaded Successfully!")

    return _model


def generate_embeddings(texts):
    """
    Generate embeddings for document chunks.
    """

    if not texts:
        return []

    model = get_model()

    print("\n========================================")
    print("GENERATING DOCUMENT EMBEDDINGS")
    print("========================================")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def generate_query_embedding(query):
    """
    Generate embedding for a user's question.

    Uses the same embedding model as the documents.
    """

    if not query:
        return None

    model = get_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    return embedding.tolist()