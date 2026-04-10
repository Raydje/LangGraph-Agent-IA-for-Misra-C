from google import genai

from app.config import get_settings

settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key)

for model in client.models.list():
    if model.supported_actions and "generateContent" in model.supported_actions:
        print(f"Model ID: {model.name}")
        print(f"Display Name: {model.display_name}")
        print("-" * 20)
