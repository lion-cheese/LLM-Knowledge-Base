from knowledgebase import PDFKnowledgeBase
from knowledgebase import DOCUMENT_SOURCE_DIRECTORY

# Create an instance of PDFKnowledgeBase using the document source directory
kb = PDFKnowledgeBase(pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY)
# Run the ingestion pipeline to load, split, and index your PDFs
kb.initiate_document_ingestion_pipeline()
