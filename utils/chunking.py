def create_chunks(text, chunk_size=700, overlap=100):
    """
    Split PDF text into overlapping chunks.

    Smaller overlapping chunks generally give the retriever
    more precise passages to work with.
    """

    if not text:
        return []

    # Clean unnecessary whitespace
    text = " ".join(text.split())

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks