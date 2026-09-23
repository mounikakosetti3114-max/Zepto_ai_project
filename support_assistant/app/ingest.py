import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = []

for file in os.listdir("docs"):
    
    path = f"docs/{file}"
    
    loader = TextLoader(path)
    
    
    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

vectordb=Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="chroma_db"
)

vectordb.persist()

print("Embedding completed")
    