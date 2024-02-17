import streamlit as st
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import getpass
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


def generate_response(url, question, openai_api_key):
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(url,),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(openai_api_type=openai_api_key))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0)


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

openai_api_key = st.sidebar.text_input('OpenAI API Key')

with st.form('my_form'):
  url = st.text_area('Enter URL:', example_url)
  question = st.text_area('Enter Question:', example_question)
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    with get_openai_callback() as cb:
        generate_response(url, question, openai_api_key)
        st.info(cb)


# st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )

# st.write(response)



# st.write("# Welcome to Streamlit! 👋")

# st.sidebar.success("Select a demo above.")

# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#       forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#       Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )

