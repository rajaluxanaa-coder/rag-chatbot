# 🤖 RAG-Powered Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your documents. Built with **Streamlit**, **LangChain**, and **Hugging Face**.

## ✨ Features

- 📄 Upload TXT documents
- 🔍 Semantic search using embeddings
- 💬 Ask questions about your documents
- 🏠 Runs 100% locally (no API keys needed!)
- 🔒 Private and secure
- ⚡ Fast and lightweight

## 🛠️ Tech Stack

- **Python 3.11+**
- **Streamlit** - Web interface
- **LangChain** - RAG pipeline
- **Hugging Face** - Embeddings model (all-MiniLM-L6-v2)
- **ChromaDB** - Vector database

## 📁 Project Structure
```

rag-chatbot/
├── streamlit_app.py      # Main app
├── rag_pipeline.py       # RAG logic
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── data/
│   └── knowledge/        # Your documents
├── vector_db/            # Vector database (auto-created)
└── README.md

```
