from src.helper import load_pdf, text_split,embedding_model
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeStore


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

pinecone = Pinecone(
    api_key=PINECONE_API_KEY
)


index_name = "medical-chatbot"

pinecone.create_index(
    name=index_name,
    dimension=1024, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

extracted_data = load_pdf('data/')
text_chunks = text_split(extracted_data)
embeddings = embedding_model()

# Chunks converted to embeddings and loaded to pinecone dataset
docsearch = PineconeStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
