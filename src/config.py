import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama # Uncomment if using Ollama

# Load environment variables
load_dotenv(override=True)

def get_llm(role: str):
    """
    Factory to switch models based on agent role.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if role == "listing":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)

    # computationally expensive
    # elif role == "listing": 
    #   return ChatOllama(model="llama3", temperature=0.7)

    elif role == "qa":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=api_key)
    elif role == "manager":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    else:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=api_key)

  