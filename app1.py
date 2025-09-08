import os
import streamlit as st
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# Fix asyncio loop
# -----------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -----------------------------
# Load API key safely
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ API_KEY not found. Set it in Render environment variables.")
os.environ["GOOGLE_API_KEY"] = api_key

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Ayurvedic Recommender", page_icon="🌿")
st.title("🌿 Ayurvedic Medicine Recommender")
st.markdown("Enter your symptoms or health condition below to get Ayurvedic recommendations.")

# -----------------------------
# Initialize per-user memory safely
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# LangChain model and embeddings
# -----------------------------
model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=0.2)
parser = StrOutputParser()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# -----------------------------
# Prompt template
# -----------------------------
template = PromptTemplate(
    template='''
You are an expert Ayurvedic practitioner and consultant.

CONVERSATION HISTORY:
{history}

CONTEXT INFORMATION:
{context}

PATIENT QUERY:
{question}

Answer:
''',
    input_variables=['context', 'question', 'history']
)

# -----------------------------
# Vector store loader (persistent)
# -----------------------------
@st.cache_resource
def load_vector_store():
    DB_DIR = "./Ayucare_db"
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        return Chroma(
            collection_name='data',
            embedding_function=embeddings,
            persist_directory=DB_DIR
        )
    else:
        with st.spinner("Building Ayurvedic knowledge base..."):
            loader = DirectoryLoader(path='data', glob='*.pdf', loader_cls=PyPDFLoader)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=40,
                separators=["\n\n", "\n", " ", ""]
            )
            text_docs = splitter.split_documents(docs)

            vector_store = Chroma.from_documents(
                documents=text_docs,
                embedding=embeddings,
                collection_name='data',
                persist_directory=DB_DIR
            )
            vector_store.persist()
            return vector_store

vector_store = load_vector_store()
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

# -----------------------------
# Memory helper function
# -----------------------------
def get_history_text():
    history = st.session_state.get("chat_history", [])
    return "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history])

# -----------------------------
# LangChain chain with memory
# -----------------------------
qa_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "history": lambda x: get_history_text()  # safe lambda
    }
    | template
    | model
    | parser
)

# -----------------------------
# User input
# -----------------------------
user_query = st.text_input("Describe your health concern (e.g., bloating, acne, hair fall):")

if user_query:
    with st.spinner("Generating Ayurvedic recommendation..."):
        response = qa_chain.invoke(user_query)

        # Save to memory safely
        st.session_state.chat_history.append({"user": user_query, "assistant": response})

        # Display response
        st.markdown("### Ayurvedic Recommendation")
        st.write(response)

# -----------------------------
# Display conversation history
# -----------------------------
if st.session_state.chat_history:
    st.markdown("## 🧠 Conversation History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Ayurvedic Recommender:** {chat['assistant']}")
        st.markdown("---")
