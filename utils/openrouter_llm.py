from config import client, MODEL_NAME


MAX_TOKENS = 2000
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
                ]
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
        # Normal successful response
        # ----------------------------------------------------

        if answer and answer.strip():

            print("\nAnswer generated successfully.")

            yield answer.strip()
            return

        # ----------------------------------------------------
        # Fallback request
        # ----------------------------------------------------

        print("\nFirst response contained no final answer.")
        print("Trying fallback request...")

        fallback_prompt = f"""
Answer the following healthcare question directly.

Use ONLY the provided healthcare information.

Give a clear and concise final answer.
Do not provide hidden reasoning or analysis.
Return ONLY the answer for the user.

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
            max_tokens=2000,

            stream=False,

            extra_body={
                "models": [
                    MODEL_NAME,
                    "openrouter/free"
                ]
            }
        )

        if not fallback_response.choices:
            raise RuntimeError(
                "OpenRouter fallback returned no choices."
            )

        fallback_choice = fallback_response.choices[0]

        print(
            "Fallback finish reason:",
            fallback_choice.finish_reason
        )

        fallback_message = fallback_choice.message
        fallback_answer = fallback_message.content

        if fallback_answer and fallback_answer.strip():

            print("\nFallback answer generated successfully.")

            yield fallback_answer.strip()
            return

        raise RuntimeError(
            "OpenRouter returned an empty answer "
            "after fallback attempt."
        )

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