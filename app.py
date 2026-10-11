import gradio as gr
from rag_pipeline import RAGChatbot
import time

# Initialize chatbot (loads once)
chatbot = RAGChatbot()

def respond(message, history):
    """Handle chat messages"""
    try:
        # Get response
        result = chatbot.query(message)
        answer = result["answer"]
        
        # Add source info
        sources = len(result["source_documents"])
        if sources > 0:
            answer += f"\n\n📚 Sources: {sources} documents referenced"
        
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

def process_file(file):
    """Handle file uploads"""
    if file is not None:
        # Save and process the file
        import shutil
        import os
        
        # Save to data/knowledge
        os.makedirs("./data/knowledge", exist_ok=True)
        file_path = f"./data/knowledge/{file.name}"
        shutil.copy(file.name, file_path)
        
        # Add to vector store
        chatbot.load_documents()
        chunks = chatbot.split_documents(chatbot.load_documents())
        chatbot.create_vector_store(chunks)
        
        return "✅ Document added successfully!"
    return ""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Chatbot") as demo:
    gr.Markdown("""
    # 🤖 RAG-Powered AI Chatbot
    ### Ask questions based on your documents (100% FREE, no API key needed!)
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot_interface = gr.ChatInterface(
                fn=respond,
                title="Chat with your documents",
                description="Ask anything about your uploaded documents"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Upload Documents")
            file_upload = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                file_count="single"
            )
            upload_status = gr.Textbox(label="Status", interactive=False)
            
            file_upload.upload(
                process_file,
                inputs=file_upload,
                outputs=upload_status
            )
            
            gr.Markdown("""
            ### 💡 Tips
            - Upload PDF documents
            - Ask questions naturally
            - Responses are from your documents
            - No internet needed after loading!
            """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")