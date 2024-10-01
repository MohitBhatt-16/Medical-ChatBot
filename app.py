from src.helper import embedding_model
from langchain_community.vectorstores import Pinecone as PineconeStore
from dotenv import load_dotenv
from src.prompt import *
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os
from flask import Flask, render_template, jsonify, request
from langchain_groq import ChatGroq
from pinecone import Pinecone


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

embeddings = embedding_model()

pinecone = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

docsearch = PineconeStore.from_existing_index(index_name,embeddings)

PROMPT = PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs = {"prompt":PROMPT}

llm = ChatGroq(model="mixtral-8x7b-32768")

qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = docsearch.as_retriever(search_kwargs = {'k':2}),
    return_source_documents = True,
    chain_type_kwargs = chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods = ["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query":input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__=='__main__':
    app.run(debug = True)

