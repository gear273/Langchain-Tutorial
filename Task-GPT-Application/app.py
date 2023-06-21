# Importing libraries
from apikey import OPENAI_API_TOKEN

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

import streamlit as st

# The language model we're going to use to control the agent
llm = OpenAI(openai_api_key=OPENAI_API_TOKEN, temperature=0.5)

# Streamlit framework
st.title("🦜🔗 Task GPT Application")

with st.sidebar:
    st.info("This application allows you to do a wide range of tasks.")
    option = st.radio('Choose your task', ['Base Gen', 'Creative', 'Summarization', 'Few Shot', 'Python'])

if option == 'Base Gen': 
    st.info('Use this application to perform standard chat generation tasks.')
    
    # Prompt box 
    prompt = st.text_input('Plug in your prompt here!')
    template = PromptTemplate(input_variables=['action'], template="""
            As a creative agent, {action}
    """)
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        # do this
        st.write(response)    

    
if option == 'Creative': 
    st.info('Use this application to perform creative tasks like writing stories and poems.')
    
    # Prompt box 
    prompt = st.text_input('Plug in your prompt here!')
    template = PromptTemplate(input_variables=['action'], template="""
            As a creative agent, {action}
    """)
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        # do this
        st.write(response)

if option == 'Summarization': 
    st.info('Use this application to perform summarization on blocks of text.')

    # Prompt box 
    prompt = st.text_area('Plug in your prompt here!')
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a passage to summarize. Using the prompt, provide a summarized response. 
            ### Prompt: 
            {action}
            ### Summary:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        # do this
        st.write(response)

if option == 'Few Shot': 
    
    st.info('Pass through some examples of task-output to perform few-shot prompting.')
    # Examples for few shots 
    examples = st.text_area('Plug in your examples!')
    prompt = st.text_area('Plug in your prompt here!')

    template = PromptTemplate(input_variables=['action','examples'], template="""
        ### Instruction: 
        The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
        ### Examples: 
        {examples}
        ### Prompt: 
        {action}
        ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(examples=examples, action=prompt) 
        print(response)
        # do this
        st.write(response)

if option == 'Python': 
    st.info('Leverage a Python agent by using the PythonREPLTool inside of Langchain.')
    # Python agent
    python_agent = create_python_agent(llm=llm, tool=PythonREPLTool(), verbose=True)
    # Prompt text box
    prompt = st.text_input('Plug in your prompt here!')
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = python_agent.run(prompt) 

        # do this
        st.write(response)