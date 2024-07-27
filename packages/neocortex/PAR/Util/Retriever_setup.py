from CustomHelper.load_model import get_openai_embedding_model
import os
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.storage.mongodb import MongoDBStore
from langchain.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pip install chromadb langchain langchain-huggingface langchain-chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_core.embeddings import Embeddings
from chromadb.api.types import EmbeddingFunction, Documents


class LangChainEmbeddingAdapter(EmbeddingFunction[Documents]):
    def __init__(self, ef: Embeddings):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        # LC EFs also have embed_query but Chroma doesn't support that so we just use embed_documents
        # TODO: better type checking
        return self.ef.embed_documents(input)


client = chromadb.PersistentClient(path="./db")

db = Chroma(
    client=client,
    collection_name="test",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
)

retriever = db.as_retriever(search_kwargs={"filter": {"id": "1"}})

load_dotenv()

mongoDBURI = os.getenv("MONGODB_URI")
mongoDB_name = os.getenv("MONGODB_NAME")
mongoDB_collection = os.getenv("MONGODB_COLLECTION")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")


mongodb_store = MongoDBStore(
    mongoDBURI, db_name=mongoDB_name, collection_name=mongoDB_collection
)

# This will be update soon!
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100)
parent_retriever = ParentDocumentRetriever(
    vectorstore=db,
    docstore=mongodb_store,
    child_splitter=child_splitter,
)
