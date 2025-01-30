import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_community.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter

st.subheader('Summarize Text')

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
source_text = st.text_area("Source Text", label_visibility="collapsed", height=250)

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    if not openai_api_key.strip() or not source_text.strip():
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner('Please wait...'):
              text_splitter = CharacterTextSplitter()
              texts = text_splitter.split_text(source_text)
              docs = [Document(page_content=t) for t in texts[:3]]

              llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
              chain = load_summarize_chain(llm, chain_type="map_reduce")
              summary = chain.run(docs)

              st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
