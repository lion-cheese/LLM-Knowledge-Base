from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Set up the callback manager with streaming output
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = OllamaLLM(model="mistral", callback_manager=callback_manager)

# Loop to interact with the model
while True:
    question = input("Ask me a question: ")
    
    if question.lower() == 'exit':
        print("Exiting...")
        break
    
    llm(question)
    print("\n__________________________________________________________________________________________\n")
