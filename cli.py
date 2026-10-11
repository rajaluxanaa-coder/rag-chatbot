from rag_pipeline import RAGChatbot
import sys

def main():
    print("🤖 RAG Chatbot (FREE - Hugging Face)")
    print("Type 'exit' to quit\n")
    
    chatbot = RAGChatbot()
    
    # Initialize
    try:
        chatbot.create_vector_store(load_existing=True)
        print("✓ Chatbot ready!")
    except:
        print("⚠️ No documents found. Add PDFs to ./data/knowledge/")
        print("   Creating new vector store...")
        chatbot.create_vector_store()
        print("✓ Ready!")
    
    while True:
        question = input("\n💬 You: ")
        if question.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye!")
            break
        
        print("🤖 Thinking...")
        response = chatbot.query(question)
        print(f"\n🤖 Bot: {response['answer']}")
        print(f"📚 Sources: {len(response['source_documents'])} documents")

if __name__ == "__main__":
    main()