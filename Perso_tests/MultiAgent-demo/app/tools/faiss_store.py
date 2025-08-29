import os
import faiss
import pickle
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDINGS = HuggingFaceEmbeddings(model_name=DEFAULT_MODEL)

def save_to_faiss(docs, save_path: str):
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
    vectorstore.save_local(save_path)

def load_faiss(path: str) -> FAISS:
    return FAISS.load_local(path, EMBEDDINGS, allow_dangerous_deserialization=True)
