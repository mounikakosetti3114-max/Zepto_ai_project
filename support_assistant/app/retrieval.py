from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


db = Chroma(persist_directory = "chroma_db", embedding_function = embedding_model)

def retrieve(query):
    
    docs = db.similarity_search(query , k=3)
    
    return docs