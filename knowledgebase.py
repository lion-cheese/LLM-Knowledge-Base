import os
from typing import Optional
from chromadb.config import Settings
from langchain_chroma import Chroma
# from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader, CSVLoader
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

    # def load_files(self):
    #     # Print all files in the directory for debugging
    #     all_files = os.listdir(self.pdf_source_folder_path)
    #     print(f"📂 Found files in directory: {all_files}")  

    #     # Use DirectoryLoader to load all file types
    #     loader = DirectoryLoader(
    #         self.pdf_source_folder_path, 
    #         glob="*",  # Load all files, not just PDFs
    #         loader_cls=UnstructuredFileLoader  # This auto-detects file types
    #     )

    #     loaded_docs = loader.load()
    #     print(f"✅ Loaded {len(loaded_docs)} documents from {self.pdf_source_folder_path}")

    #     if not loaded_docs:
    #         raise ValueError("❌ ERROR: No files were loaded! Ensure the directory contains valid files.")

    #     return loaded_docs
    
    def load_files(self):
        all_files = os.listdir(self.pdf_source_folder_path)
        print(f"📂 Found files in directory: {all_files}")  

        loaded_docs = []
        
        for file in all_files:
            file_path = os.path.join(self.pdf_source_folder_path, file)
            
            # Choose the right loader
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")  # Ensure proper encoding
            elif file.endswith(".csv"):
                loader = CSVLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)  # Fallback for other types

            try:
                docs = loader.load()
                print(f"✅ Loaded {len(docs)} documents from {file}")

                # Debug: Print the first 500 characters of extracted text
                for doc in docs:
                    print(f"\n🔍 Extracted Preview from {file}:\n{doc.page_content[:500]}\n{'-'*80}")

                loaded_docs.extend(docs)

            except Exception as e:
                print(f"❌ ERROR loading {file}: {e}")

        if not loaded_docs:
            raise ValueError("❌ ERROR: No valid documents were loaded!")

        return loaded_docs
    
    def split_documents(self, loaded_docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],  # Ensures natural breaks
            length_function=len,  # Ensures words aren't cut mid-sentence
        )
        
        chunked_docs = splitter.split_documents(loaded_docs)

        # Debug: Print some chunk previews
        # for i, doc in enumerate(chunked_docs[:5]):  # Print first 5 chunks
        #     print(f"\n📝 Chunk {i+1} Preview:\n{doc.page_content[:500]}\n{'-'*80}")

        # print(f"✅ Created {len(chunked_docs)} document chunks.")

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
