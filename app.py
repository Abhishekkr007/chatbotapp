import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import ui
import pdf_processing
import requests  # Used to call the FastAPI endpoint

# Load environment variables and configure the Google API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for chat history and app start flag
if "messages" not in st.session_state:
    st.session_state.messages = []
if "started" not in st.session_state:
    st.session_state.started = False

def get_chatbot_response(question):
    """Call the FastAPI endpoint to get the chatbot response."""
    url = "http://127.0.0.1:8000/query"
    payload = {"question": question}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "")
        else:
            return "I'm sorry, I couldn't find an answer."
    except Exception as e:
        return f"Error: {str(e)}"
def main():
    """Main function to run the Streamlit app."""
    if not st.session_state.started:
        display_welcome_page()
    else:
        # Link the external CSS file
        st.markdown('<link rel="stylesheet" href="assets/style.css">', unsafe_allow_html=True)

        ui.load_css()
        ui.display_intro()
        
        st.header("🤖 Q & A: Your AI Assistant")

        st.markdown("""**Hello! I am your AI assistant.** 🔍  
        I can help you process and extract information from your documents.

        **How to Get Started:**
        1. Upload your PDF documents.
        2. Click on the "Process Documents" button to analyze the content.
        3. Ask me any question about your documents, and I'll provide relevant information.

        Let's explore your documents together! 🚀
        """)

        # File uploader for PDFs
        pdf_docs = st.file_uploader("Upload PDFs for analysis", type=["pdf"], accept_multiple_files=True)

        # Process Documents button
        process_button = st.button("Process Documents", help="Click to process the uploaded PDFs!")
        if process_button:
            if pdf_docs:
                with st.spinner("Processing your PDFs... 🧐"):
                    raw_text = pdf_processing.get_pdf_text(pdf_docs)
                    text_chunks = pdf_processing.get_text_chunks(raw_text)
                    pdf_processing.get_vector_store(text_chunks)
                    st.success("Documents processed successfully! You can now ask questions about your documents.")
            else:
                st.warning("Please upload at least one PDF.")

        st.markdown("""### "Your Docs 🔍 My Answers." """)
        
        # Display chat history: User + Assistant response in the same container
        for i in range(0, len(st.session_state.messages), 2):
            with st.container():
                user_msg = st.session_state.messages[i]["content"]
                assistant_msg = st.session_state.messages[i + 1]["content"] if i + 1 < len(st.session_state.messages) else "..."
                st.markdown(f"""
                <div class="chat-bubble">
                    <div class="user-message">🧑‍💻 <strong>You:</strong> {user_msg}</div>
                    <div class="assistant-message">🤖 <strong>Assistant:</strong> {assistant_msg}</div>
                </div>
                """, unsafe_allow_html=True)

        # Chat input
        user_question = st.chat_input("Ask me anything about your documents!")
        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            response = get_chatbot_response(user_question)
            if not response:
                response = "I'm sorry, I couldn't find an answer to your question."
            
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Re-render the page to show the new messages
            st.rerun()

        # End Chat button
        if st.session_state.messages:
            if st.button("End Chat", help="Click to clear chat history and restart session!", key="end_chat_btn"):
                st.session_state.messages = []
                st.session_state.started = False
                st.rerun()
                # Display 'End' with custom styling
                st.markdown('<span class="end-label">End</span>', unsafe_allow_html=True)

def display_welcome_page():
    """Displays the welcome page to introduce the user to the app."""
    st.title("Welcome to Q & A: Your AI Assistant")
    st.markdown("""**Welcome! I am here to assist you in extracting and analyzing information from your documents.**  
    Upload your PDFs and start exploring their content with ease.

    Click the "Get Started" button to begin.
    """)
    if st.button("Get Started"):
        st.session_state.started = True

if __name__ == "__main__":
    main()
