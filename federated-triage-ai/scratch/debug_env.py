import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load from the same directory as this file
env_path = Path(__file__).parent.parent / '.env'
print(f"DEBUG: Looking for .env at: {env_path}")
load_dotenv(dotenv_path=env_path)

openai_key = os.environ.get("API_KEY", "NOT_FOUND")
gemini_key = os.environ.get("GEMINI_API_KEY", "NOT_FOUND")

print(f"API_KEY: {openai_key[:15]}...{openai_key[-5:] if len(openai_key) > 5 else ''}")
print(f"CWD: {os.getcwd()}")
