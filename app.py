import streamlit as st
from ragmodel import query_greek_with_rag

# Set Streamlit app title and description
st.title("Greek Mythology Assistant")
st.write("Ask questions about Greek mythology and get comprehensive answers from our collection!")

# Input field for user queries
query = st.text_input("Enter your question about Greek mythology:", "")

# Button to trigger the query
if st.button("Get Answer"):
    if query.strip():
        try:
            st.spinner("Fetching the answer, please wait...")
            response = query_greek_with_rag(query, collection_name="collection_el", n_results=5)
            st.subheader("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
    else:
        st.warning("Please enter a valid question!")

# Footer
st.write("---")
st.write("Powered by **ChromaDB** and **Meltemi**")
