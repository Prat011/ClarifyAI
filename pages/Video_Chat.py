import streamlit as st
import os
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.youtube_transcript.utils import is_youtube_video
import qdrant_client
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.readers.web import FireCrawlWebReader
from llama_index.core import SummaryIndex
import streamlit_analytics2 as streamlit_analytics
import time
import dotenv

dotenv.load_dotenv()
# Set page config
#st.set_page_config(page_title="Talk to Youtube Video", page_icon="📚", layout="wide")

# Initialize session state
if 'setup_complete' not in st.session_state:
    st.session_state['setup_complete'] = False
if 'documents' not in st.session_state:
    st.session_state['documents'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'index' not in st.session_state:
    st.session_state['index'] = None
if 'url' not in st.session_state:
    st.session_state['url'] = ""
if 'collection_name' not in st.session_state:
    st.session_state['collection_name'] = ""
if 'query' not in st.session_state:
    st.session_state['query'] = ""

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Setup functions
def embed_setup():
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(temperature=0.1, model_name="models/gemini-pro")

def qdrant_setup():
    client = qdrant_client.QdrantClient(
      os.getenv("QDRANT_URL"),
      api_key = os.getenv("QDRANT_API_KEY"),
    )
    return client

def llm_setup():
    llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1, model_name="models/gemini-pro")
    return llm

def query_index(index, streaming=True):
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            """You are an AI assistant for developers, specializing in Video Summarization. Your task is to provide accurate, concise, and helpful responses based on the given documentation context.
                Context information is below:
                {context_str}
                Always answer based on the information in the context and general knowledge and be precise
                Given this context, please respond to the following user query:
                {query_str}
                Your response should:
                Directly address the query using information from the context
                Include relevant code examples or direct quotes if applicable
                Mention specific sections or pages of the documentation
                Highlight any best practices or potential pitfalls related to the query
                After your response, suggest 3 follow-up questions based on the context that the user might find helpful for deeper understanding.
                Your response:"""
        ),
    )
    return chat_engine

# Document ingestion function
def ingest_documents(url):
    loader = YoutubeTranscriptReader()

    if is_youtube_video(url): 
        documents = loader.load_data(
            ytlinks=[url]
        )
        return documents 
    else:
        st.error("Link not supported unfortunately, the link should follow the format: <https://youtube.com/watch?v={video_id}> ")

    

# Streamlit app
st.title("Talk to any Youtube Video")

st.markdown("""
This tool allows you to chat with Video Content. Here's how to use it:
1. Enter the URL of the Youtube Video you want to chat about (optional if using an existing collection).
2. Enter the collection name for the vector store.
3. Click the "Ingest and Setup" button to crawl the documentation (if URL provided) and set up the query engine.
4. Once setup is complete, enter your query in the text box.
5. Click "Search" to get a response based on the documentation.
6. View your chat history in the sidebar.
""")

with streamlit_analytics.track():
    # URL input for document ingestion
    st.session_state['url'] = st.text_input("Enter URL to crawl and ingest documents (optional):", value=st.session_state['url'])
    
    # Collection name input
    st.session_state['collection_name'] = st.text_input("Enter collection name for vector store:", value=st.session_state['collection_name'])
    
    # Combined Ingest and Setup button
    if st.button("Ingest and Setup"):
        with st.spinner("Setting up query engine..."):
            embed_setup()
            client = qdrant_setup()
            llm = llm_setup()
            vector_store = QdrantVectorStore(client=client, collection_name=st.session_state['collection_name'])
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            if st.session_state['url']:
                st.session_state['documents'] = ingest_documents(st.session_state['url'])
                st.session_state['index'] = VectorStoreIndex.from_documents(st.session_state['documents'], vector_store=vector_store, storage_context=storage_context)
                st.success(f"Documents ingested from {st.session_state['url']} and query engine setup completed successfully!")
            else:
                st.session_state['index'] = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
                st.success(f"Query engine setup completed successfully using existing collection: {st.session_state['collection_name']}")
            
            st.session_state['setup_complete'] = True
    
    # Query input
    st.session_state['query'] = st.text_input("Enter your query:", value=st.session_state['query'])
    
    # Search button
    if st.button("Search"):
        if not st.session_state['setup_complete']:
            st.error("Please complete the setup first")
        elif st.session_state['query']:
            with st.spinner("Searching..."):
                try:
                    chat_engine = query_index(st.session_state['index'])
                    response = chat_engine.chat(st.session_state['query'])
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Retrying in 120 seconds...")
                    time.sleep(120)
                    try:
                        chat_engine = query_index(st.session_state['index'])
                        response = chat_engine.chat(st.session_state['query'])
                    except Exception as e:
                        st.error(f"Retry failed. Error: {str(e)}")
                        st.stop()

            # Add the query and response to chat history
            st.session_state['chat_history'].append(("User", st.session_state['query']))
            st.session_state['chat_history'].append(("Assistant", str(response.response)))
            
            # Display the most recent response prominently
            st.subheader("Assistant's Response:")
            st.write(response.response)
        else:
            st.error("Please enter a query")
    
    # Sidebar for chat history
    st.sidebar.title("Chat History")
    for role, message in st.session_state['chat_history']:
        st.sidebar.text(f"{role}: {message}")
    
    # Clear chat history button in sidebar
    if st.sidebar.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.sidebar.success("Chat history cleared!")