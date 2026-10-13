
import os
from pathlib import Path
from dotenv import load_dotenv

# Find the .env file in the main project folder
project_folder = Path(__file__).resolve().parent.parent
env_path = project_folder / ".env"

load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("SUCCESS: Gemini API key was loaded.")
else:
    print("ERROR: API key not found. Check your .env file.")
