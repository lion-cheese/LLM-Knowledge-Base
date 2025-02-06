import os
from typing import Optional
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.document_loaders import DirectoryLoader  # Updated import
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants for Chroma vector database and document processing
CHROMA_DB_DIRECTORY = 'db'
DOCUMENT_SOURCE_DIRECTORY = '/path/to/source/documents'
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=CHROMA_DB_DIRECTORY,
    anonymized_telemetry=False
)
TARGET_SOURCE_CHUNKS = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HIDE_SOURCE_DOCUMENTS = False

class PDFKnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        self.pdf_source_folder_path = pdf_source_folder_path
    
    def load_pdfs(self):
        loader = DirectoryLoader(self.pdf_source_folder_path)
        loaded_pdfs = loader.load()
        return loaded_pdfs

    def split_documents(self, loaded_docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs
    
    def convert_document_to_embeddings(self, chunked_docs, embedder):
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedder,
            client_settings=CHROMA_SETTINGS
        )
        vector_db.add_documents(chunked_docs)
        vector_db.persist()
        return vector_db
    
    def return_retriever_from_persistant_vector_db(self, embedder):
        if not os.path.isdir(CHROMA_DB_DIRECTORY):
            raise NotADirectoryError("Please load your vector database first.")
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedder,
            client_settings=CHROMA_SETTINGS
        )
        return vector_db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
    
    def initiate_document_injetion_pipeline(self):
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)
        print("=> PDF loading and chunking done.")
        embeddings = OllamaEmbeddings()
        vector_db = self.convert_document_to_embeddings(chunked_docs=chunked_documents, embedder=embeddings)
        print("=> Vector DB initialized and created.")
        print("All done")
