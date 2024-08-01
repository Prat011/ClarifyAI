import os
import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import DocumentSummaryIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import google.generativeai as genai
import os
import PyPDF2
import streamlit_analytics2 as streamlit_analytics


# Set up Google API key

# Configure Google Gemini
Settings.embed_model = GeminiEmbedding(api_key=os.getenv("GOOGLE_API_KEY"), model_name="models/embedding-001")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.8, model_name="models/gemini-pro")
llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1, model_name="models/gemini-pro")

# Load and index the input data
def load_data(document_text):
    document = [Document(text=document_text)]
    
    #index = VectorStoreIndex.from_documents([document])
    index = VectorStoreIndex.from_documents(document)
    return index

# Default report format template
DEFAULT_REPORT_FORMAT = """
    Title Page
        Includes the report title, author's name, and date.
    Abstract
        A concise summary of the report, covering the background, objectives, methodology, key findings, and conclusions.
    Table of Contents
        Lists sections and subsections with corresponding page numbers for easy navigation.
    Introduction
        Provides background information, defines the scope of the report, and states the objectives.
    Literature Review
        Reviews relevant literature and previous research related to the report topic.
    Methodology/Approach
        Details the methods used to gather data or conduct experiments, including design and analytical techniques.
    Results and Discussion
        Presents findings in a clear format, often using tables, figures, and charts, followed by a discussion interpreting these results.
    Conclusions
        Summarizes the main findings and their implications, often linking back to the report's objectives.
    Recommendations
        Suggests actions based on the findings, highlighting potential future work or improvements.
    References
        Lists all sources cited in the report, adhering to a specific referencing style.
    Appendices
        Contains supplementary material that supports the main text, such as raw data, detailed calculations, or additional figures.
"""

# Generate report
def generate_report(index, report_format, additional_info):
    query_engine = index.as_query_engine()
    
    if not report_format.strip():
        report_format = DEFAULT_REPORT_FORMAT
        st.info("Using default report format.")
    
    response = query_engine.query(f"""
    You are a professional report writer. Your task is to create a comprehensive report based on the entire document provided.
    
    First, thoroughly analyze and summarize the entire document. Then, use the input text to create a well-structured report following the format below:
    
    Report Format:
    {report_format}
    
    Additional Information:
    {additional_info}
    
    Even if the input is shallow, generate a report
    Guidelines:
    1. Ensure you comprehend and summarize the entire document before starting the report.
    2. The report should be comprehensive, covering all major points from the document.
    3. Adapt the provided format as necessary to best fit the content and context of the document.
    4. Incorporate any additional information provided into the relevant sections of the report.
    5. Use clear, professional language throughout the report.
    6. Provide specific examples or data from the document to support your analysis and conclusions.
    7. If the document contains technical information, explain it in a way that's accessible to a general audience.
    
    Generate a thorough, well-structured report that captures the essence of the entire document.
    """)
    return response.response

# Streamlit app
def main():
    st.title("AI Report Writer")
    st.write("Upload your document and our AI will generate a comprehensive report based on its contents!")

    with streamlit_analytics.track():

    # File uploader
        uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=["txt", "pdf"])
        
        # Report format input
        report_format = st.text_area("Enter the desired report format (optional)", height=150, 
                                     help="Leave blank to use a default template")
        
        # Additional information input
        additional_info = st.text_area("Enter any additional information or context for the report", height=100)
    
        if uploaded_file is not None:
            # Read file contents
            if uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                document_text = ""
                for page in pdf_reader.pages:
                    document_text += page.extract_text()
            else:
                document_text = uploaded_file.getvalue().decode("utf-8")
    
            if st.button("Generate Report"):
                st.write("Analyzing document and generating report...")
    
                # Load data and generate report
                doc_list = document_text.split(".")
                index = load_data(document_text)
                report = generate_report(index, report_format, additional_info)
    
                st.write("## Generated Report")
                st.write(report)
        else:
            st.warning("Please upload a file to generate a report.")

if __name__ == "__main__":
    main()