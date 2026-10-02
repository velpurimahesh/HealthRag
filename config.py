import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found.\n\n"
        "Please create a .env file in the project folder "
        "and add:\n\n"
        "OPENROUTER_API_KEY=your_api_key_here"
    )

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
    max_retries=0
)

MODEL_NAME = "inclusionai/ling-3.0-flash-sante:free"