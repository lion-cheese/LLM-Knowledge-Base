from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from knowledgebase import PDFKnowledgeBase, DOCUMENT_SOURCE_DIRECTORY
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings

# Set up a callback manager for streaming output.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Instantiate the LLM with the "mistral" model hosted locally.
llm = OllamaLLM(
    model="mistral",
    api_base="http://localhost:11434",
    streaming=True,
    callback_manager=callback_manager
)

# Instantiate the embeddings object.
embeddings = OllamaEmbeddings()

# Create an instance of your knowledge base using the consistent parameter name.
kb = PDFKnowledgeBase(pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY)

# Retrieve the retriever from your persistent vector database.
retriever = kb.return_retriever_from_persistant_vector_db(embedder=embeddings)

# Create the RetrievalQA chain to connect the LLM with the retriever.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

print("Enter your queries (type 'exit' to quit):")
while True:
    query = input("What's on your mind: ")
    if query.lower() == 'exit':
        break

    result = qa_chain(query)
    answer = result['result']
    docs = result['source_documents']

    print("\nAnswer:")
    print(answer)
    print("\n" + "#" * 30 + " Source " + "#" * 30)

    for document in docs:
        source = document.metadata.get("source", "Unknown")
        print(f"\n> SOURCE: {source}:")
        print(document.page_content)
    print("#" * 70 + "\n")
