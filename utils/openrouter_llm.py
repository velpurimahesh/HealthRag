from config import client, MODEL_NAME


MAX_TOKENS = 1000
TEMPERATURE = 0.1


def generate_answer(prompt):

    print("\n" + "=" * 60)
    print("OPENROUTER - GENERATING ANSWER")
    print("=" * 60)

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,

            stream=False,

            extra_body={
                "models": [
                    MODEL_NAME,
                    "openrouter/free"
                ],

                "reasoning": {
                    "effort": "none"
                }
            }
        )

        print("\nModel:", response.model)

        if not response.choices:
            raise RuntimeError(
                "OpenRouter returned no choices."
            )

        choice = response.choices[0]

        print(
            "Finish reason:",
            choice.finish_reason
        )

        message = choice.message

        answer = message.content

        # ----------------------------------------------------
        # IMPORTANT:
        # Some models may return reasoning but no content.
        # Do NOT display reasoning as the final answer.
        # ----------------------------------------------------

        if answer and answer.strip():

            print("\nAnswer generated successfully.")

            yield answer.strip()
            return

        # ----------------------------------------------------
        # If content is empty, try once more with a simpler
        # request that strongly requires a final answer.
        # ----------------------------------------------------

        print("\nFirst response contained no final answer.")
        print("Trying fallback request...")

        fallback_prompt = f"""
Answer the following healthcare question directly.

Use ONLY the provided healthcare information.

Do not provide reasoning or analysis.
Do not explain your thinking.
Return ONLY the final answer.

Question and context:

{prompt}
"""

        fallback_response = client.chat.completions.create(
            model=MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content": fallback_prompt
                }
            ],

            temperature=0.0,
            max_tokens=1000,

            stream=False,

            extra_body={
                "models": [
                    MODEL_NAME,
                    "openrouter/free"
                ],

                "reasoning": {
                    "effort": "none"
                }
            }
        )

        if not fallback_response.choices:
            raise RuntimeError(
                "OpenRouter fallback returned no choices."
            )

        fallback_message = (
            fallback_response
            .choices[0]
            .message
        )

        fallback_answer = fallback_message.content

        if not fallback_answer or not fallback_answer.strip():

            raise RuntimeError(
                "OpenRouter returned an empty answer "
                "after fallback attempt."
            )

        print("\nFallback answer generated successfully.")

        yield fallback_answer.strip()

    except Exception as e:

        print("\n" + "=" * 60)
        print("OPENROUTER ERROR")
        print("=" * 60)

        print(
            "ERROR TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            repr(e)
        )

        print("=" * 60)

        yield (
            "Sorry, I could not generate an answer "
            "right now. Please try your question again."
        )