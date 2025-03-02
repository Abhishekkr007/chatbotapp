import streamlit as st

import streamlit as st



def load_css():
    """Function to load custom CSS for styling."""
    try:
        with open("style.css", "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file 'style.css' not found. Please ensure it exists in the app directory.")

def display_intro():
    """Displays an introduction message."""
    st.markdown('<p class="big-font">Welcome! I am here to assist you in extracting and analyzing information from your documents.</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Upload your PDFs and start exploring their content with ease.</p>', unsafe_allow_html=True)

