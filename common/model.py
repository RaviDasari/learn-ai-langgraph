import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_model(model_name=None, temperature=0):
    """
    Returns a configured ChatOpenAI instance.
    
    Args:
        model_name (str, optional): The name of the model to use. Defaults to env var MODEL_NAME or "gpt-4o".
        temperature (float, optional): The temperature to use. Defaults to 0.
        
    Returns:
        ChatOpenAI: The configured model instance.
    """
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        
    return ChatOpenAI(
        model=model_name,
        temperature=temperature
    )
