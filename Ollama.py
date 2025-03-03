import streamlit as st
import ollama
import os
import nltk
from chromadb.config import Settings

# Importing necessary LangChain modules
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Define the LLM model
model = "llama3.2"

def dataProcess(file_Upload):
    """
    Processes the uploaded PDF file and loads it as a document.
    :param file_Upload: Uploaded PDF file
    :return: Loaded document object
    """
    try:
        loader = UnstructuredPDFLoader(file_path=file_Upload)
        document = loader.load()
        print("Processing successful")
        return document
    except Exception as e:
        print("Error loading document:", e)
        return None

def spitTxt(document):
    """
    Splits the document into smaller text chunks for better processing.
    :param document: Loaded document object
    :return: List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Define the chunk size
        chunk_overlap=300,  # Overlapping text to maintain context
    )
    try:
        chunks = text_splitter.split_documents(document)
        return chunks
    except Exception as e:
        print("Error splitting document:", e)
        return None

def vectorDB(chunks):
    """
    Creates a vector database from document chunks using embeddings.
    :param chunks: List of document chunks
    :return: Vector database
    """
    try:
        # Pull the embedding model
        ollama.pull('nomic-embed-text')
        
        # Create and store the embeddings
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
        )
        print("Done Embedding")
        return vectordb
    except Exception as e:
        print("Error creating vector database:", e)
        return None

def main():
    """
    Main function to handle Streamlit app functionality.
    """
    st.title("AI Document Reader")
    
    # File uploader widget
    df = st.file_uploader(label="Upload a PDF File")
    st.write("Waiting for file upload...")
    
    if df is not None:
        # Process uploaded document
        data = dataProcess(df)
        if data is not None:
            st.write("Processing file...")
            split = spitTxt(data)
            if split is not None:
                # Create vector database from document chunks
                vectordb = vectorDB(split)
                
                if vectordb is not None:
                    userInput = st.text_input("Input Your Question")
                    if userInput:
                        # Initialize the language model
                        llm_instance = ChatOllama(
                            model=model,
                            temperature=0.8,
                            num_predict=256,
                        )
                        
                        # Define prompt template for multi-query retrieval
                        prompt_template = PromptTemplate(
                            input_variables=["question"],
                            template="""You are an AI language model assistant. Your task is to generate five
                            different versions of the given user question to retrieve relevant documents from
                            a vector database. By generating multiple perspectives on the user question, your
                            goal is to help the user overcome some of the limitations of the distance-based
                            similarity search. Provide these alternative questions separated by newlines.
                            Original question: {question}""",
                        )
                        
                        # Create a MultiQueryRetriever for better document search
                        retriever_from_llm = MultiQueryRetriever.from_llm(
                            retriever=vectordb.as_retriever(),
                            llm=llm_instance,
                            prompt=prompt_template,
                        )
                        
                        # Define the response template
                        template = """Answer the question based ONLY on the following context:
                        {context}
                        Question: {question}
                        """
                        
                        chat_prompt = ChatPromptTemplate.from_template(template=template)
                        
                        # Create the processing chain
                        chain = (
                            {"context": retriever_from_llm, "question": RunnablePassthrough()}
                            | chat_prompt
                            | llm_instance
                            | StrOutputParser()
                        )
                        
                        # Execute the retrieval and generation process
                        result = chain.invoke(userInput)
                        
                        # Display the result
                        st.write("### Answer:")
                        st.write(result)
                    else:
                        st.write("Please enter a question to query the database.")
                else:
                    st.write("Failed to create vector database.")
            else:
                st.write("Failed to split the document into chunks.")
        else:
            st.write("Failed to process the file. Please upload a valid document.")
    else:
        st.write("Please upload a document to start.")

if __name__ == "__main__":
    main()
