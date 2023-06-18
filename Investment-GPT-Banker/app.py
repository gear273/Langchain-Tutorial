# Importing libraries
from apikey import HUGGINGFACEHUB_API_TOKEN

from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

import streamlit as st

# Streamlit framework
st.title("🦜🔗 Investment GPT Banker")
input_text = st.text_input("Input your prompt here")

# The language model we're going to use to control the agent
llm = HuggingFaceHub(
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
    repo_id = "bigscience/bloom-560m", 
    model_kwargs = {"temperature": 0.9, "max_length": 250}
)
embeddings = HuggingFaceHubEmbeddings(
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN
)

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