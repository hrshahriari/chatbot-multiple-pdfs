import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

# Path to the 'knowledge' folder
KNOWLEDGE_FOLDER = 'knowledge'

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_file_text(txt_docs):
    text = ""
    for txt in txt_docs:
        content = txt.read().decode('utf-8')  # Decode bytes to string
        if content.strip():  # Check if the file is not empty
            text += content + "\n"
        else:
            st.warning(f"The text file '{txt.name}' is empty.")
    return text

def get_csv_text(csv_docs):
    text = ""
    for csv in csv_docs:
        try:
            df = pd.read_csv(csv)  # Read CSV file into a DataFrame
            if not df.empty:  # Check if the DataFrame is not empty
                text += df.to_string(index=False) + "\n"  # Convert to string, without index
            else:
                st.warning(f"The CSV file '{csv.name}' is empty.")
        except Exception as e:
            st.error(f"Error reading CSV file '{csv.name}': {e}")
    return text

def read_knowledge_files():
    raw_text = ""
    for filename in os.listdir(KNOWLEDGE_FOLDER):
        file_path = os.path.join(KNOWLEDGE_FOLDER, filename)
        if filename.endswith('.pdf'):
            with open(file_path, 'rb') as pdf_file:
                raw_text += get_pdf_text([pdf_file])  # Process PDF files
        elif filename.endswith('.txt'):
            with open(file_path, 'rb') as txt_file:
                raw_text += get_text_file_text([txt_file])  # Process text files
        elif filename.endswith('.csv'):
            try:
                raw_text += get_csv_text([file_path])  # Process CSV files
            except Exception as e:
                st.error(f"Error reading CSV file '{filename}': {e}")

    return raw_text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):  
    if st.session_state.conversation is None:  
        st.warning("Please process documents first to start the conversation.")  
        return  

    response = st.session_state.conversation({'question': user_question})  
    st.session_state.chat_history = response['chat_history']  

    for i, message in enumerate(st.session_state.chat_history):  
        if i % 2 == 0:  
            st.write(user_template.replace(  
                "{{MSG}}", message.content), unsafe_allow_html=True)  
        else:  
            st.write(bot_template.replace(  
                "{{MSG}}", message.content), unsafe_allow_html=True)  

    # Clear the user input field  
    st.session_state.user_input = ""  # Clear the input text field

def main():  
    load_dotenv()  
    st.set_page_config(page_title="I'm your drinking assistant! How can I help you",  
                       page_icon="🧋")  
    st.write(css, unsafe_allow_html=True)  

    if "conversation" not in st.session_state:  
        st.session_state.conversation = None  
    if "chat_history" not in st.session_state:  
        st.session_state.chat_history = []  
    if "user_input" not in st.session_state:  # Initialize the user input  
        st.session_state.user_input = ""  

    st.header("Drinks assistant 🧋")  

    # Initially read the knowledge files and process them  
    raw_text = read_knowledge_files()  
    if raw_text:  
        # Get the text chunks  
        text_chunks = get_text_chunks(raw_text)  

        # Create vector store  
        vectorstore = get_vectorstore(text_chunks)  

        # Create the conversation chain  
        st.session_state.conversation = get_conversation_chain(vectorstore)  

    # Create a text input box for user questions  
    user_question = st.text_input("Ask me:", value=st.session_state.user_input)  

    # Handle user input  
    if user_question and st.session_state.conversation:  
        handle_userinput(user_question)  

    with st.sidebar:  
        st.subheader("Options")  
        
        # New chat button  
        if st.button("New Chat"):  
            st.session_state.chat_history = []  # Clear chat history  
            st.session_state.user_input = ""  # Also clear input text  
            st.success("Chat history cleared. You can continue asking questions.")

if __name__ == '__main__':
    main()