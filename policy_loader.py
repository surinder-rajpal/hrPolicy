from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "./chroma"
embeddings_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def loadAndSplitPoilicies(path: str):
    documents = []

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs = loader.load()

            for d in docs:
                documents.append(d)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings_function,
        persist_directory=CHROMA_DIR
    )
    return vectordb

def get_retriever():
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings_function)
    return vectordb.as_retriever(search_kwargs={"k": 3})
