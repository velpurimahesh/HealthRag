import os
import chromadb


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "healthcare_knowledge"


def get_chroma_client():

    os.makedirs(CHROMA_PATH, exist_ok=True)

    return chromadb.PersistentClient(
        path=CHROMA_PATH
    )


def get_collection():

    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )


def reset_collection():

    client = get_chroma_client()

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        pass

    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )


def store_in_chromadb(
    chunks,
    embeddings,
    metadatas=None
):

    if not chunks:
        raise ValueError(
            "No chunks available for ChromaDB."
        )

    collection = reset_collection()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    if metadatas is None:

        metadatas = [
            {
                "source": "healthcare_document"
            }
            for _ in chunks
        ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )

    return collection