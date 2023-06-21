# Importing libraries
from apikey import OPENAI_API_TOKEN

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

import streamlit as st

# Streamlit framework
st.title("🦜🔗 Investment GPT Application")
input_text = st.text_input("Input your prompt here")

# The language model we're going to use to control the agent
llm = OpenAI(openai_api_key=OPENAI_API_TOKEN, temperature=0.1)
embeddings = OpenAIEmbeddings()

# Create and load PDF loader
loader = PyPDFLoader("data/annualreport.pdf")  

# Split pages from pdf 
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name = "annual_report",
    description = "a banking annual report as a pdf",
    vectorstore = store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm = llm,
    toolkit = toolkit,
    verbose = True
)

# Application
if input_text:
    response = agent_executor.run(input_text)

    st.write(response)

    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(input_text) 
        # Write out the first 
        st.write(search[0][0].page_content) 