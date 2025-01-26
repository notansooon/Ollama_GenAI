
import streamlit as st
import os
import ollama 

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
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')        
        

model = "llama3.2"



def dataProcess(data):
        if data:
                print("Data Exist:", data)
        else:
                print("error")
                return None
        

        try:
                loader = UnstructuredPDFLoader(file_path=data)
                document = loader.load()
                print("Processing successful")
                
                content = document[0].page_content
                
                return document


        except Exception as e:
                print(f"Error during processing: {e}")
                return None


        







def spitTxt(document):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
        )
        str(document)
        try: 
                chunks = text_splitter.split_documents(document)
                return chunks
        except Exception as e:
                print("error here")
                print("Error", {e})
        print("Done spliiting") 

        

def vectorDB(chunks):
        ollama.pull('nomic-embed-text')

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model='nomic-embed-text')
        )
        print("Done Embedding")

        return vectordb

#Vector Db


def retrieval(model, vectordb):
    llm = ChatOllama(
        model=model,
        temperature=0.8,
        num_predict=256,
    )

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(),
        llm=llm,
        prompt=prompt_template,
    )

    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template=template)

    chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )



def main():
        st.title("AI Document Reader")

        
        df = st.file_uploader(label="File Upload")
               


        data =  dataProcess(df)

        if (data is None):
                st.write("Failed to process the file. Please upload a valid document.")
        

        split = spitTxt(data)

        st.text_input("Input Your Question")




if __name__ == "__main__":
   main()