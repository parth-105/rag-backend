from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pinecone
import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone
pc = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
# CORS(app)

def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def sanitize_user_id(user_id):
    sanitized_id = re.sub(r'[^a-z0-9-]', '-', user_id.lower())
    print(f"Sanitized user_id: {sanitized_id}")  # Add logging to debug
    return sanitized_id

def get_vector_store(text_chunks, user_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    sanitized_user_id = sanitize_user_id(user_id)
    index_name = f"user-{sanitized_user_id}-index"
    dimension = 768  # Set the dimension manually based on the model
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=pinecone.ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    vector_store = Pinecone.from_texts(text_chunks, embedding=embeddings, index_name=index_name)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant. Answer the question as detailed as possible using the provided context. If the answer is not in the provided context, say, "The answer is not available in the context provided."

    Context: {context}

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, user_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    sanitized_user_id = sanitize_user_id(user_id)
    index_name = f"user-{sanitized_user_id}-index"
    vector_store = Pinecone(index_name=index_name, embedding=embeddings)
    docs = vector_store.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        file = request.files['file']
        user_id = request.form['user_id']
        raw_text = get_pdf_text(file)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks, user_id)
        return jsonify({"message": "PDF processed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']
        user_id = data['user_id']
        answer = user_input(question, user_id)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
