import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Document
import google.generativeai as genai
import streamlit_analytics2 as streamlit_analytics

# Set up Google API key
import os

# Configure Google Gemini

# Load and index the resume data
def load_data(uploaded_files):
    documents = [Document(text=t) for t in uploaded_files]
    #documents = SimpleDirectoryReader(input_files=[uploaded_files]).load_data()
    Settings.embed_model = GeminiEmbedding(api_key=os.getenv("GOOGLE_API_KEY"), model_name="models/embedding-001")
    Settings.llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.8, model_name="models/gemini-pro")
    llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1, model_name="models/gemini-pro")
    index = VectorStoreIndex.from_documents(documents)
    return index

# Generate resume feedback
def generate_feedback(index, resume_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(f"""
    You are a Standup Comedian, Your job is to roast the input given to you.
    Be Extremely FUNNY, use various Joke structures including one liners, setup punchline
    Analyze the following resume and roast it:
    {resume_text}
    
    Please cover the following aspects:
    1. Overall impression
    2. Format and structure
    3. Content quality
    4. Areas for improvement
    
    """)
    return response.response

# Streamlit app

def main():
    st.title("Resume Roaster")
    st.write("Upload a resume, and let our AI roast it!")
    with streamlit_analytics.track():
    # File uploader
        uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf"])

    if uploaded_file is not None:
        # Read file contents
        if uploaded_file.type == "application/pdf":
            # You'll need to install PyPDF2 for this
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            resume_text = ""
            l=[]
            for page in pdf_reader.pages:
                
                resume_text += page.extract_text()
                l.append(page.extract_text())
        else:
            resume_text = uploaded_file.getvalue().decode("utf-8")

        st.write("Analyzing resume...")

        # Load data and generate feedback
        index = load_data(l)
        feedback = generate_feedback(index, resume_text)

        st.write("## Resume Feedback")
        st.write(feedback)

if __name__ == "__main__":
    main()