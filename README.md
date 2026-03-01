# 🤖 CV RAG Assistant using Mistral AI & Streamlit

A Retrieval-Augmented Generation (RAG) web app that transforms Abrar’s CV into an intelligent AI assistant capable of answering questions strictly based on resume content.

---

## 🚀 Features

📄 Reads CV from a local `.txt` file  
🧠 Uses vector embeddings + FAISS for semantic search  
🤖 Powered by Mistral AI (`mistral-large-latest`)  
💬 Modern chat-style UI built with Streamlit  
⚡ Persistent vector database (no re-embedding on refresh)  
🔒 Answers strictly limited to CV content  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| LLM | Mistral AI |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | FAISS |
| RAG Framework | LangChain (modular version) |
| Environment | Python 3.11 |
