import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

@st.cache_data
def load_pdf_from_streamlit(pdf_data):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_data)
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    os.unlink(temp_file_path)

    return pages