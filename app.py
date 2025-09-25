from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# ---------------------------
# Flask app init
# ---------------------------
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENROUTER_API_KEY=os.environ.get('OPENROUTER_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# Safety check
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY is missing. Add it to your .env file.")



# ---------------------------
# Embeddings + Pinecone
# ---------------------------
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ---------------------------
# Chat model (xAI Grok via OpenRouter)
# ---------------------------
chatModel = ChatOpenAI(
    model="x-ai/grok-4-fast:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User: {msg}")
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

# ---------------------------
# Run app
# ---------------------------
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)

