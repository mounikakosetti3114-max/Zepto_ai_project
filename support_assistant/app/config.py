import os
from dotenv import load_dotenv

load_dotenv()

MOCK_LLM = os.getenv("MOCK_LLM" , "1")