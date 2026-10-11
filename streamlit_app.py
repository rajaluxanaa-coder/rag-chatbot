import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 RAG-Powered Chatbot")
st.markdown("### Ask questions based on your documents")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📂 Document Upload")
    st.markdown("---")
    
    # Upload section
    st.subheader("Upload Your Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload documents to ask questions about"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        # Create directory
        os.makedirs("./data/knowledge", exist_ok=True)
        
        # Save file
        file_path = f"./data/knowledge/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("🔄 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Clear existing vector store to reload fresh
                    if os.path.exists("./vector_db"):
                        import shutil
                        shutil.rmtree("./vector_db")
                    
                    # Load all documents
                    documents = []
                    
                    # Load PDFs
                    pdf_files = [f for f in os.listdir("./data/knowledge") if f.endswith('.pdf')]
                    if pdf_files:
                        pdf_loader = DirectoryLoader(
                            "./data/knowledge",
                            glob="**/*.pdf",
                            loader_cls=PyPDFLoader
                        )
                        documents.extend(pdf_loader.load())
                        st.info(f"📄 Found {len(pdf_files)} PDF(s)")
                    
                    # Load Text files
                    txt_files = [f for f in os.listdir("./data/knowledge") if f.endswith('.txt')]
                    if txt_files:
                        txt_loader = DirectoryLoader(
                            "./data/knowledge",
                            glob="**/*.txt",
                            loader_cls=TextLoader,
                            loader_kwargs={'encoding': 'utf-8'}
                        )
                        documents.extend(txt_loader.load())
                        st.info(f"📄 Found {len(txt_files)} text file(s)")
                    
                    if documents:
                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50
                        )
                        chunks = text_splitter.split_documents(documents)
                        
                        # Create vector store
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=st.session_state.embeddings,
                            persist_directory="./vector_db",
                            collection_name="knowledge_base"
                        )
                        st.session_state.vector_store.persist()
                        st.session_state.documents_loaded = True
                        
                        st.success(f"✅ Processed {len(documents)} document(s)!")
                        st.info(f"📊 Created {len(chunks)} searchable chunks")
                    else:
                        st.error("❌ No documents found to process. Please upload valid PDF or TXT files.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Try uploading a simple text file first to test.")
    
    # Show current documents
    st.markdown("---")
    st.subheader("📚 Loaded Documents")
    
    if os.path.exists("./data/knowledge"):
        files = os.listdir("./data/knowledge")
        if files:
            for file in files:
                st.write(f"📄 {file}")
        else:
            st.info("No documents uploaded yet")
    else:
        st.info("No documents uploaded yet")
    
    # Clear buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All"):
            if os.path.exists("./vector_db"):
                import shutil
                shutil.rmtree("./vector_db")
            st.session_state.vector_store = None
            st.session_state.documents_loaded = False
            st.session_state.messages = []
            st.rerun()

# ========== MAIN CHAT AREA ==========
st.subheader("💬 Chat with Your Documents")

# Show status
if st.session_state.documents_loaded:
    st.success("✅ Documents are loaded and ready!")
else:
    st.warning("⚠️ Please upload and process a document in the sidebar first")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        if not st.session_state.documents_loaded:
            response = "⚠️ Please upload and process a document first!"
        elif st.session_state.vector_store is None:
            response = "⚠️ No documents loaded. Please upload a document."
        else:
            with st.spinner("🔍 Searching for relevant information..."):
                try:
                    # Search for relevant documents
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    
                    if docs:
                        response = "📚 **Found relevant information:**\n\n"
                        for i, doc in enumerate(docs, 1):
                            response += f"**Document {i}:**\n{doc.page_content}\n\n"
                        response += f"\n*✨ Found {len(docs)} relevant sections in your documents*"
                    else:
                        response = "😕 No relevant information found in your documents."
                except Exception as e:
                    response = f"❌ Error: {str(e)}"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Upload a PDF or TXT file in the sidebar, then ask questions about it!")