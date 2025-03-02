import streamlit as st
import google.generativeai as genai
import pdf_processing

def user_input(query):
    """Handles user queries and fetches AI-generated responses based on document content."""
    
    # Retrieve stored document data
    vector_store = pdf_processing.get_stored_vector_store()

    if not vector_store:
        st.warning("No processed documents found. Please upload and process documents first.")
        return

    # Generate AI response
    response = generate_response(query, vector_store)

   
    st.markdown("### 🤖 AI Response")
    st.write(response)

def generate_response(query, vector_store):
    """Generates AI response based on user query and document content."""
    
    try:
        # Search for relevant document sections
        context = vector_store.similarity_search(query, k=5)
        context_text = "\n\n".join([doc.page_content for doc in context])

        # Format prompt for AI
        prompt = f"Using the following document excerpts, answer the query concisely and professionally:\n\n{context_text}\n\nUser Query: {query}"

        # Generate AI response
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        return response.text if response else "No relevant information found."

    except Exception as e:
        return f"An error occurred: {str(e)}"
