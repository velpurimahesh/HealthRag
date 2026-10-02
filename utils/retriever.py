from utils.embeddings import generate_query_embedding


def semantic_search(
    collection,
    query,
    top_k=6
):
    """
    Retrieve the most relevant healthcare passages
    from ChromaDB.
    """

    if collection is None:
        return []

    if not query:
        return []

    # Create embedding for user question
    query_embedding = generate_query_embedding(
        query
    )

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved = []

    for i, document in enumerate(documents):

        metadata = {}

        if i < len(metadatas):
            metadata = metadatas[i] or {}

        distance = None

        if i < len(distances):
            distance = distances[i]

        retrieved.append({
            "text": document,
            "source": metadata.get(
                "source",
                "Unknown document"
            ),
            "distance": distance
        })

    return retrieved