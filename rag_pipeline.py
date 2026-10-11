from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import torch
from config import Config

class RAGChatbot:
    def __init__(self):
        self.config = Config()
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL
        )
        
        print("🔄 Loading LLM (this takes 2-3 minutes first time)...")
        self.llm = self._load_huggingface_llm()
        
        self.vector_store = None
        self.qa_chain = None
        
    def _load_huggingface_llm(self):
        """Load free Hugging Face model"""
        model_name = self.config.LLM_MODEL
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def load_documents(self, data_path="./data/knowledge"):
        """Load documents from directory"""
        loader = DirectoryLoader(
            data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        return documents
    
    def split_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vector_store(self, chunks=None, load_existing=False):
        """Create or load vector store"""
        import os
        if load_existing and os.path.exists(self.config.CHROMA_PERSIST_DIR):
            self.vector_store = Chroma(
                persist_directory=self.config.CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=self.config.COLLECTION_NAME
            )
            return self.vector_store
        
        if chunks is None:
            docs = self.load_documents()
            chunks = self.split_documents(docs)
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.config.CHROMA_PERSIST_DIR,
            collection_name=self.config.COLLECTION_NAME
        )
        self.vector_store.persist()
        return self.vector_store
    
    def query(self, question):
        """Query the chatbot"""
        if self.vector_store is None:
            self.create_vector_store(load_existing=True)
        
        # Search for relevant documents
        docs = self.vector_store.similarity_search(question, k=self.config.TOP_K_RESULTS)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response,
            "source_documents": docs
        }

# For CLI usage
class StreamingRAGChatbot(RAGChatbot):
    def query_stream(self, question):
        """Stream responses"""
        response = self.query(question)
        yield response["answer"]