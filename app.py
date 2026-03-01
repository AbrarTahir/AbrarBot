import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ----------------------------
# CONFIG
# ----------------------------

CV_PATH = "abrar_tahir_cv.txt"
DB_PATH = "faiss_index"

load_dotenv()

st.set_page_config(
    page_title="Abrar AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# CUSTOM STYLING
# ----------------------------

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------------
# SIDEBAR
# ----------------------------

with st.sidebar:
    st.title("⚙️ Control Panel")

    st.markdown("### 📄 CV Status")
    if os.path.exists(CV_PATH):
        st.success("CV Loaded")
    else:
        st.error("CV file missing")

    st.markdown("---")

    if st.button("🔄 Rebuild Knowledge Base"):
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            st.success("Index deleted. Restart the app.")
        else:
            st.info("No index found.")

    st.markdown("---")
    st.markdown("Built with:")
    st.markdown("- Mistral AI")
    st.markdown("- FAISS Vector DB")
    st.markdown("- Streamlit")


# ----------------------------
# HEADER
# ----------------------------

st.title("🤖 Abrar's AI CV Assistant")
st.caption("Ask anything about Abrar based strictly on his CV.")

st.divider()


# ----------------------------
# VECTOR STORE LOADING
# ----------------------------

@st.cache_resource
def load_vectorstore():

    if not os.path.exists(CV_PATH):
        st.error("CV file not found.")
        st.stop()

    with open(CV_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    documents = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(
            DB_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )
    else:
        with st.spinner("🔍 Building knowledge base..."):
            vectorstore = FAISS.from_documents(docs, embedding)
            vectorstore.save_local(DB_PATH)

    return vectorstore


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

llm = ChatMistralAI(model="mistral-large-latest")


# ----------------------------
# CHAT MEMORY
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ----------------------------
# CHAT INPUT
# ----------------------------

if prompt := st.chat_input("Ask about Abrar's skills, education, experience..."):

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            relevant_docs = retriever.invoke(prompt)

            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )

            final_prompt = f"""
You are an AI assistant answering questions STRICTLY based on the CV content below.

If the answer is not present in the CV, respond with:
"I cannot find that information in the CV."

CV Context:
{context}

Question:
{prompt}
"""

            response = llm.invoke(final_prompt)

            answer = response.content
            st.markdown(answer)

    # Store assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )