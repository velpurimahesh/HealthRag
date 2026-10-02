import os

from utils.pdf_reader import read_pdf
from utils.chunking import create_chunks
from utils.embeddings import generate_embeddings
from utils.vector_db import store_in_chromadb


DATA_FOLDER = "data"


def build_knowledge_base():

    print("\n========================================")
    print("HEALTHRAG KNOWLEDGE BASE BUILDER")
    print("========================================\n")

    if not os.path.exists(DATA_FOLDER):

        print("ERROR: data folder not found.")
        return


    pdf_files = [
        file
        for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(".pdf")
    ]


    if not pdf_files:

        print("ERROR: No PDF files found.")
        return


    print(
        f"Found {len(pdf_files)} healthcare PDFs.\n"
    )


    all_chunks = []
    all_metadatas = []


    # ==========================================
    # READ PDFs
    # ==========================================

    for index, filename in enumerate(pdf_files):

        print(
            f"[{index + 1}/{len(pdf_files)}] "
            f"Reading: {filename}"
        )

        pdf_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        try:

            text = read_pdf(pdf_path)

            chunks = create_chunks(
                text,
                chunk_size=700,
                overlap=100
            )


            for chunk in chunks:

                all_chunks.append(chunk)

                all_metadatas.append({
                    "source": filename
                })


            print(
                f"    → {len(chunks)} chunks"
            )


        except Exception as e:

            print(
                f"    ERROR: {e}"
            )


    # ==========================================
    # CHECK
    # ==========================================

    if not all_chunks:

        print(
            "\nERROR: No text was extracted."
        )

        return


    print("\n========================================")
    print(
        f"TOTAL CHUNKS: {len(all_chunks):,}"
    )
    print("========================================\n")


    # ==========================================
    # EMBEDDINGS
    # ==========================================

    print(
        "Creating embeddings..."
    )

    embeddings = generate_embeddings(
        all_chunks
    )


    # ==========================================
    # CHROMADB
    # ==========================================

    print(
        "\nSaving embeddings to ChromaDB..."
    )

    store_in_chromadb(
        all_chunks,
        embeddings,
        all_metadatas
    )


    print("\n========================================")
    print("KNOWLEDGE BASE READY ✅")
    print("========================================")

    print(
        "\nYou can now run:"
    )

    print(
        "streamlit run app.py"
    )


if __name__ == "__main__":

    build_knowledge_base()