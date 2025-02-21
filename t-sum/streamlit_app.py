import configparser
import os
import streamlit as st
from langchain_openai import OpenAI

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '.', 'config.ini'))
openai_api_key = config['OPENAI'].get('apikey')

def generate_response(txt):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)

    docs = [ Document(page_content=t) for t in texts ]
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.invoke({"input_documents": docs})

# Page title
st.set_page_config(page_title='Text Summariser')
st.title('Text Summariser')
st.caption("<p style='color:white'> A summarisation app powered by Streamlit, OpenAI and Langchain. <p>", unsafe_allow_html=True)

# Text input
txt_to_summarise = st.text_area('Enter your text', '', height=400)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Processing...'):
                    response = generate_response(txt_to_summarise)["output_text"]
                    result.append(response)

if len(result):
    st.info(response)
