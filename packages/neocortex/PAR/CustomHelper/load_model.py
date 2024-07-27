import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


# def get_cohere_model(model_name="command-r", temperature=0) -> ChatCohere:
#     llm = ChatCohere(
#         model=model_name,
#         temperature=temperature
#     )
#     return llm


def get_openai_model(temperature=0.3) -> ChatOpenAI:
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
        max_tokens=4096,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return llm


def get_openai_embedding_model() -> OpenAIEmbeddings:
    """Load the OpenAI embedding model easily. You can freely revise it to make it easier to use."""
    embedding = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="http://localhost:1234/v1",
    )
    return embedding
