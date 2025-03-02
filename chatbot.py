import os
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Download NLTK resources
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")  # You can choose a larger model for better accuracy

# Initialize Hugging Face QA pipeline (using a popular model)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Initialize NLTK stemmer
stemmer = PorterStemmer()

def preprocess_text_spacy_nltk(text):
    """
    Preprocesses the text by tokenizing with spaCy, removing stopwords with NLTK, and performing named entity recognition using spaCy.
    """
    # Process the text with spaCy
    doc = nlp(text)

    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha and token.text.lower() not in stop_words]
    
    # Stemming using NLTK's PorterStemmer
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Perform Named Entity Recognition (NER) with spaCy
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Join tokens back into a cleaned text
    cleaned_text = " ".join(stemmed_tokens)
    
    return cleaned_text, entities

def get_conversational_chain():
    prompt_template = """
    Answer the user's question concisely and clearly, based on the provided context.
    If relevant information is not found, respond with: "I'm sorry, but I do not have enough information to answer this question."

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, use_hf=False):
    """
    Handle user queries and retrieve answers from the FAISS vector store.
    
    Parameters:
      user_question (str): The user's query.
      use_hf (bool): If True, use the Hugging Face QA pipeline; otherwise, use the generative chain.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Ensure FAISS index exists
    if not os.path.exists("faiss_index") or not os.path.isfile("faiss_index/index.faiss"):
        return "No processed documents found. Please upload and process documents first."

    # Load FAISS index and perform similarity search
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    if not docs:
        return "I'm sorry, but no relevant information was found in the documents."

    # Combine retrieved document texts into one context.
    context = " ".join([doc.page_content for doc in docs if hasattr(doc, "page_content")]) or " ".join(docs)

    # Preprocess the context using both spaCy and NLTK (tokenization, stopword removal, NER, stemming)
    cleaned_context, entities = preprocess_text_spacy_nltk(context)

    if use_hf:
        try:
            hf_result = qa_pipeline(question=user_question, context=cleaned_context)
            hf_answer = hf_result.get("answer", "")
            return hf_answer
        except Exception as e:
            return f"Error with Hugging Face QA: {str(e)}"
    else:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]

