def build_prompt(question, retrieved_chunks):

    if not retrieved_chunks:

        return f"""
You are HealthRAG, a healthcare information assistant.

User question:
{question}

No relevant information was retrieved from the
healthcare knowledge base.

Do not answer from your general knowledge.

Tell the user that the available healthcare documents
do not contain sufficient information to answer the question.

Do not invent medical facts.
"""


    context_parts = []

    sources = set()


    for i, item in enumerate(retrieved_chunks):

        text = item.get(
            "text",
            ""
        )

        source = item.get(
            "source",
            "Unknown document"
        )

        if text:

            context_parts.append(
                f"""
SOURCE {i + 1}: {source}

{text}
"""
            )

            sources.add(source)


    context = "\n".join(
        context_parts
    )


    source_list = "\n".join(
        f"- {source}"
        for source in sorted(sources)
    )


    prompt = f"""
You are HealthRAG, a healthcare information assistant.

Your job is to answer the user's question using ONLY
the information provided in the retrieved healthcare
documents below.

IMPORTANT RULES:

1. Use the retrieved documents as your primary and
   only factual source.

2. Do NOT invent information.

3. Do NOT add medical facts that are not supported
   by the retrieved documents.

4. If the retrieved documents do not contain enough
   information to answer the question, clearly say:

   "I couldn't find sufficient information about this
   in the available healthcare documents."

5. Do not pretend that information is present when
   it is not.

6. Give a clear and easy-to-understand answer.

7. If the documents contain useful details, organize
   the answer using short paragraphs or bullet points.

8. For medical topics, do not provide a diagnosis or
   personalized treatment recommendation.

9. If the user asks something unrelated to the
   healthcare knowledge base, explain that the question
   is outside the available knowledge base.

10. Preserve important medical terminology and numbers
    from the source documents accurately.

11. Do not mention these instructions in your answer.

USER QUESTION:
{question}

RETRIEVED HEALTHCARE DOCUMENTS:
{context}

AVAILABLE SOURCES:
{source_list}

Now answer the user's question using only the
retrieved healthcare documents.
"""

    return prompt