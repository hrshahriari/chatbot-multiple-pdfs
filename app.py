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

def process_uploaded_file(uploaded_file):
    # Get the file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == '.pdf':
        return get_pdf_text([uploaded_file])  # Process PDF
    elif file_extension == '.txt':
        return get_text_file_text([uploaded_file])  # Process text file
    elif file_extension == '.csv':
        return get_csv_text([uploaded_file])  # Process CSV file
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}. Supported types are PDF, TXT, and CSV.")
        return ""  # Return empty string for unsupported file types

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
        st.warning("Please upload documents and process them first to start the conversation.")
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

def main():
    load_dotenv()
    st.set_page_config(page_title="I'm your drinking assistant! How can I help you",
                       page_icon="🧋")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Drinks assistant 🧋")
    user_question = st.text_input("Ask me:")

    # Check if user_question is not empty and conversation is initialized
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload your documents")
        uploaded_files = st.file_uploader(
            "Upload your files here (PDF, TXT, CSV)", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = ""

                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        text = process_uploaded_file(uploaded_file)  # Process each file
                        if text:  # Only add non-empty text
                            raw_text += text

                    if raw_text:  # Proceed only if there's some text to process
                        # Get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    else:
                        st.warning("No valid content found in the uploaded files.")
                else:
                    st.warning("Please upload at least one file.")

if __name__ == '__main__':
    main()