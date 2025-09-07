from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableMap, RunnablePassthrough
import os

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')


template = PromptTemplate(
    template='''
You are an Ayurvedic practitioner and consultant. Provide accurate, safe, and easy-to-understand Ayurvedic recommendations in a **short and concise way**.

CONTEXT INFORMATION:
{context}

PATIENT QUERY:
{question}

INSTRUCTIONS:
- Keep the answer **short and simple**, use plain language.  
- Focus on the **main issue and 2–3 practical remedies only**.  
- Always mention **precautions** clearly.  
- If the condition is **severe, long-lasting, recurring, or risky (e.g., pregnancy, chronic illness, chest pain, high fever, diabetes, etc.)**, then add this disclaimer at the end:

"IMPORTANT MEDICAL DISCLAIMER: These Ayurvedic tips are for educational purposes only. If your symptoms persist, worsen, or if you have a serious medical condition, consult a qualified healthcare professional or Ayurvedic doctor immediately."

RESPONSE FORMAT:
- **Problem Summary**: 1–2 lines.  
- **Remedy**: Herbs / home remedies (short + specific).  
- **Precautions**: When to avoid, safety warnings.  
- Add disclaimer **only if condition is severe**.
''',
    input_variables=['context', 'question']
)


parser=StrOutputParser()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
DB_DIR = "./Ayucare_db"

if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    vector_store = Chroma(
        collection_name='data',
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    print("Vector store loaded from disk")
else:
    print("Building vector store...")

loader = DirectoryLoader(
        path='data',
        glob='*.pdf',
        loader_cls=PyPDFLoader
)
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
print(" Vector store built and saved")


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)


qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | template
    | model
    | StrOutputParser()
)

# query=input('enter your health concerns')
# response = qa_chain.invoke(query)
# print(response)
while True:
        query=input('enter your health concerns')
        if query=='exit' or query=='bye':
            break
        else:
          response = qa_chain.invoke(query)
          print(response)

