import streamlit as st
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from linkpreview import link_preview, LinkGrabber
import os

# for compatibility with chromaDB versioning
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# for Langsmith tracing of prompts
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# To use .env for langsmith and openaiAPI Keys
import dotenv
dotenv.load_dotenv()

st.set_page_config(
    page_title="Web QA",
    page_icon="🕸️",
    layout="wide"
)


def generate_response(url, question):
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(url,),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.session_state.openai_api_key, temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    st.info(response)


example_url = 'https://medium.com/@jesse.henson/get-answers-directly-from-documents-using-chatgpt-f02e83592caf'
example_question = 'Summarize this article'

with st.sidebar.form("API Key"):
    openai_api_key = st.text_input('OpenAI API Key')
    submitted = st.form_submit_button('Add Key')
    if submitted:
        if openai_api_key.startswith('sk-'):
            st.session_state.openai_api_key = openai_api_key
        else: 
            st.sidebar.warning('Enter a different Key', icon='⚠')
st.title("🕸️ Web QA")
st.session_state.url = st.text_input('Enter URL:', example_url)


col1, col2 = st.columns([1,1], gap="large")



with col1:
    with st.form('my_form'):
        question = st.text_area('Enter Question:', example_question)
        no_api_key = 'openai_api_key' not in st.session_state
        submitted = st.form_submit_button('Submit', disabled=no_api_key)
        if no_api_key:
            st.warning('Please enter your OpenAI API key!', icon='⚠')            
        if submitted and not no_api_key:
            with get_openai_callback() as cb:
                generate_response(st.session_state.url, question)
                st.info(cb)

with col2:
    if 'url' in st.session_state:
        with st.expander("Preview",True):
            preview = link_preview(st.session_state.url)
            preview_image = preview.absolute_image
            st.title(preview.title)
            st.subheader(preview.description)
            st.image(preview.image)