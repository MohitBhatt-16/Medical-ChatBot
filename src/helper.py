from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeEmbeddings
import os

# Extract data from the PDF
def load_pdf(pdf_file):
    loader = DirectoryLoader(pdf_file,
                             glob = "*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Embeddings from Pinecone
def embedding_model():
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    return embeddings