import os
from typing import Optional
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants for Chroma vector database and document processing
CHROMA_DB_DIRECTORY = 'db'
DOCUMENT_SOURCE_DIRECTORY = r"C:\\Users\\Lian Cheng\\LangChainApp\\source_documents"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False
)
TARGET_SOURCE_CHUNKS = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HIDE_SOURCE_DOCUMENTS = False

class PDFKnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        self.pdf_source_folder_path = pdf_source_folder_path

    def load_files(self):
        # Print all files in the directory for debugging
        all_files = os.listdir(self.pdf_source_folder_path)
        print(f"📂 Found files in directory: {all_files}")  

        # Use DirectoryLoader to load all file types
        loader = DirectoryLoader(
            self.pdf_source_folder_path, 
            glob="*",  # Load all files, not just PDFs
            loader_cls=UnstructuredFileLoader  # This auto-detects file types
        )

        loaded_docs = loader.load()
        print(f"✅ Loaded {len(loaded_docs)} documents from {self.pdf_source_folder_path}")

        if not loaded_docs:
            raise ValueError("❌ ERROR: No files were loaded! Ensure the directory contains valid files.")

        return loaded_docs

    def split_documents(self, loaded_docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    def convert_document_to_embeddings(self, chunked_docs, embedder):
        vector_db = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embedder,
            persist_directory=CHROMA_DB_DIRECTORY
        )
        print("✅ Vector DB initialized and saved!")
        return vector_db

    
    def return_retriever_from_persistant_vector_db(self, embedder):
        if not os.path.isdir(CHROMA_DB_DIRECTORY):
            raise NotADirectoryError("Please load your vector database first.")
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedder,
            client_settings=Settings(anonymized_telemetry=False)
        )
        return vector_db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
    
    def initiate_document_ingestion_pipeline(self):
        loaded_docs = self.load_files()  

        chunked_documents = self.split_documents(loaded_docs=loaded_docs)
        print("=> PDF loading and chunking done.")
        embeddings = OllamaEmbeddings(model="mistral")

        vector_db = self.convert_document_to_embeddings(chunked_docs=chunked_documents, embedder=embeddings)
        print("=> Vector DB initialized and created.")
        print("All done")
