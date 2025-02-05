from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from knowledgebase import MyKnowledgeBase
from knowledgebase import DOCUMENT_SOURCE_DIRECTORY
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = OllamaLLM(model="mistral", api_base="http://localhost:11434", streaming=True, callback_manager=callback_manager)
embeddings = OllamaEmbeddings()
kb = MyKnowledgeBase(pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY)
retriever = kb.return_retriever_from_persistant_vector_db(embedder=embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True, verbose=True)

print("Enter your queries (type 'exit' to quit): ")
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