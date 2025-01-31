
import streamlit as st
import ollama
import os
import tempfile
from chromadb.config import Settings

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import llm

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

import nltk

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

model = "llama3.2"

def dataProcess(file_Upload):
    
    
    
   
    loader = UnstructuredPDFLoader(file_path=file_Upload)
    document = loader.load()
    print("Processing successful")

    return document

def spitTxt(document):
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )
    try:
        chunks = text_splitter.split_documents(document)
        return chunks
    except Exception as e:
        print("Error splitting document:", e)
        return None

def vectorDB(chunks):
    # Pull the embedding model
    ollama.pull('nomic-embed-text')
    

    # Create the vector database
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
        
        
    )
    print("Done Embedding")
    return vectordb

def main():
    
    
    
    st.title("AI Document Reader")

    #df = st.file_uploader(label="File Upload")
    df = "./data/market.pdf"
    st.write("this is not running")
    if df is not None:
        
        data = dataProcess(df)
        if data is not None:
            st.write("what about here")
            split = spitTxt(data)
            if split is not None:
                vectordb = vectorDB(split)

                if vectordb is not None:
                    userInput = st.text_input("Input Your Question")
                    if userInput:
                        
                        llm_instance = ChatOllama(
                            model=model,
                            temperature=0.8,
                            num_predict=256,
                        )

                        # PromptTemplate for multi-query retrieval
                        prompt_template = PromptTemplate(
                            input_variables=["question"],
                            template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
                        )

                        # Create a MultiQueryRetriever
                        retriever_from_llm = MultiQueryRetriever.from_llm(
                            retriever=vectordb.as_retriever(),
                            llm=llm_instance,
                            prompt=prompt_template,
                        )

                        
                        template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

                        chat_prompt = ChatPromptTemplate.from_template(template=template)

                        # Create the chain
                        chain = (
                            {"context": retriever_from_llm, "question": RunnablePassthrough()}
                            | chat_prompt
                            | llm_instance
                            | StrOutputParser()
                        )

                        
                        result = chain.invoke(userInput)

                       
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
