import streamlit as st
import langchain_helper as lch
import pdf_helper as ph
import textwrap

# Streamlit app
st.title("Patricia's Pdf Assistant")

# Streamlit sidebar form
with st.sidebar:
    with st.form(key='my_form'):
        st.cache_data.clear()
        st.cache_resource.clear()
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            pdf_data = uploaded_file.getvalue()
            pages = ph.load_pdf_from_streamlit(pdf_data)
        query = st.text_area(
            label="Ask me about the PDF?",
            max_chars=150,
            key="query",
            height=None
        )
        submit_button = st.form_submit_button(label='Submit')

# Process the form submission
if query and submit_button:
   
    db = lch.get_database(pages)
    llm = lch.get_language_model()
    vectorstore_info = lch.get_vector_store_info(db)
    vector_tool = lch.get_vector_store_tools(llm, vectorstore_info)
    agent_executor = lch.get_vector_store_agent(llm, vector_tool)

    response = agent_executor.run(query)
    
    st.subheader("Answer:")
    formatted_response = textwrap.fill(response, width=85)
    st.markdown(f"<pre>{formatted_response}</pre>", unsafe_allow_html=True)
    
    response = db.similarity_search_with_score(response)
    st.expander("expand").write(response)
