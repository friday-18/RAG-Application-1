from flask import Flask, render_template, jsonify, request, session
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os
import uuid


app = Flask(__name__)

load_dotenv()

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-.env")

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


embeddings = download_embeddings()

index_name = "the-indian-penal-code-index"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Rewrites a follow-up into a standalone question before searching Pinecone
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rewrite the question "
     "so it can be understood without the history. Do NOT answer it. "
     "If it is already standalone, return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_q_prompt
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Buffer memory: one full history per session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())

    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session["sid"]}},
    )
    print("Response : ", response["answer"])
    return str(response["answer"])


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.pop("sid", None)
    if sid:
        store.pop(sid, None)
    return "", 204


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8070, debug=True)