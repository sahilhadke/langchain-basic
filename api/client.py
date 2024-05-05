import requests 
import streamlit as st

def get_response(input_text):
    response = requests.post("http://localhost:8000/api/invoke", 
    json={"input": {'topic': input_text}}
    )
    return response.json()['output']['content']


# streamlit
st.title("LangChain API")
input_text = st.text_input("Enter a topic")
if st.button("Generate"):
    output = get_response(input_text)
    st.write(output)