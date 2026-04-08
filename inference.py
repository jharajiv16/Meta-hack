import os
import requests

# ✅ REQUIRED ENV VARIABLES (IMPORTANT FOR CHECKLIST)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ❗ NO DEFAULT VALUE HERE (VERY IMPORTANT)
HF_TOKEN = os.getenv("HF_TOKEN")


def call_model(prompt):
    """
    Generic function to call LLM API
    Works with OpenAI-compatible endpoints
    """

    headers = {
        "Content-Type": "application/json",
    }

    # If token exists, add Authorization header
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    url = f"{API_BASE_URL}/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a startup advisor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ API Error: {str(e)}"


# OPTIONAL: helper for your simulator
def get_startup_advice(state):
    prompt = f"""
    You are an expert startup mentor.

    Current startup state:
    {state}

    Give 1 short actionable advice.
    """

    return call_model(prompt)