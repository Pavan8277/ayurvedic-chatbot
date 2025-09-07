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


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


load_dotenv()


st.set_page_config(page_title="Ayurvedic Recommender", page_icon="🌿")
st.title("🌿 Ayurvedic Medicine Recommender")
st.markdown("Enter your symptoms or health condition below to get Ayurvedic recommendations.")


model = ChatGoogleGenerativeAI(model='gemini-2.5-pro',temperature=0.2)
parser = StrOutputParser()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

template = PromptTemplate(
    template='''
You are an expert Ayurvedic practitioner and consultant with deep knowledge of traditional Indian medicine, herbs, remedies, and holistic healing approaches. Your role is to provide accurate, safe, and evidence-based Ayurvedic recommendations based on the patient's symptoms and condition.

CONTEXT INFORMATION:
{context}

PATIENT QUERY:
{question}

INSTRUCTIONS:
1.Assessment: Carefully analyze the patient's symptoms, constitution (Prakriti), and current imbalance (Vikriti) based on the three doshas (Vata, Pitta, Kapha).

2. Ayurvedic Diagnosis: Provide a preliminary Ayurvedic understanding of the condition, including:
   - Which dosha(s) are imbalanced
   - Possible root causes (Nidan) according to Ayurveda
   - Stage of the disease process

3. Recommendations: Provide comprehensive guidance including:
   - Herbal Remedies: Specific herbs, formulations (with proper Sanskrit names), dosage, and preparation methods
   - Dietary Guidelines: Foods to include and avoid based on the condition and dosha imbalance
   - Lifestyle Modifications: Daily routine (Dinacharya), exercise, sleep patterns, and stress management
   - Home Remedies: Simple, safe remedies using common household ingredients
   - Yoga/Pranayama: Specific asanas or breathing techniques if applicable

4. Safety Considerations:
   - Mention any potential contraindications or side effects
   - Highlight if certain herbs should be avoided during pregnancy, lactation, or with specific medical conditions
   - Note any possible interactions with modern medications

5. Dosage and Duration: Provide clear instructions on how to take recommended remedies, frequency, timing (before/after meals), and expected duration of treatment.

IMPORTANT MEDICAL DISCLAIMER:
Always conclude your response with this exact statement:

"IMPORTANT MEDICAL DISCLAIMER: While these Ayurvedic recommendations are based on traditional knowledge and practices, they are meant for educational purposes only. If your symptoms persist for more than 2 days, worsen, or if you experience any adverse reactions, please consult a qualified healthcare professional or Ayurvedic doctor immediately. Do not delay seeking professional medical attention for serious or persistent health concerns."

Reply to only medical questions if other than medical questions come reply sorry i am not trained to answer those questons
RESPONSE FORMAT:
Structure your response clearly with headings and bullet points for easy readability. Be thorough yet concise, ensuring the patient understands both the remedies and their proper application.

Answer:
''',
    input_variables=['context', 'question']
)


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


qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | template
    | model
    | parser
)

user_query = st.text_input("Describe your health concern (e.g., bloating, acne, hair fall):")

if user_query:
    with st.spinner("Generating Ayurvedic recommendation..."):
        response = qa_chain.invoke(user_query)
        st.markdown("Ayurvedic Recommendation")
        st.write(response)
