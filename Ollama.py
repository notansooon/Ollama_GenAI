
import streamlit as st
import os

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader


import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

data = "./data/test.pdf"

if  os.path.exists("./data/test.pdf"):
        print("Path exist")
else: 
        print("path does not Exist")
        
        

model = "llama3.2"

if data:
        print("Processing started for:", data)

        try:
           loader = UnstructuredPDFLoader(file_path=data)
           document = loader.load()
           print("Processing successful")
           
           content = document[0].page_content
           print(content[:100])


        except Exception as e:
           print(f"Error during processing: {e}")
else:
        print("error")


from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import llm


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
)

chunks = text_splitter.split_text(document)

print("Done spliiting")


#Vector Db
import ollama 

ollama.pull('nomic-embed-text')

vectordb = Chroma.from_documents(
      documents=chunks,
      embedding=OllamaEmbeddings(model='nomic-embed-text')
)
print("Done Embedding")


#R
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from langchain_core.runnables import Runnable
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain

llm = ChatOllama(
    model = model,
    temperature = 0.8,
    num_predict = 256,

)

prompt_template = PromptTemplate(
      input_variables=["Question"],
      template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",

)


retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm, prompt=prompt_template
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template=template)

chain = LLMChain(llm=llm, prompt=prompt_template)