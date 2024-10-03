import streamlit as st
from langchain_community.document_loaders import NotebookLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Streamlit app title
st.title("Review your Data Science Project")

# Initialize the Google Generative AI LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyCjmESx4o2-FwFusJRcdQqQvTy0Gn0Ihm0")

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your Jupyter notebook", type=["ipynb"])

# Textbox for user question
question = st.text_input("Enter your question:")

if uploaded_file:
    # Save the uploaded file to 'notebook.ipynb'
    file_path = "notebook.ipynb"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Load the notebook
    loader = NotebookLoader(
        file_path,
        include_outputs=True,
        remove_newline=True,
    )
    
    docs = loader.load()
    
    if st.button("Analyze"):
        # Define the Summarize Chain
        template = question + """Answer the question based of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        
        response1 = stuff_chain.invoke(docs)
        st.markdown("### Summary of the Jupyter Notebook Document")
        st.write(response1["output_text"])
    
else:
    st.write("Please upload a Jupyter notebook to analyze.")
