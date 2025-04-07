import os
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_API_ENBEDDINGS_DEPLOYMENT_NAME"),
)

# Create a sample knowledge base
texts = [
    "Azure OpenAI Service provides REST API access to OpenAI's powerful language models including the GPT-3, Codex and Embeddings model series.",
    "Vector search is a technique used in information retrieval and machine learning to find similar items in a large dataset.",
    "LangChain is a framework for developing applications powered by language models.",
]

# Create a vector store
vectorstore = FAISS.from_texts(texts, embeddings)

# Set up the conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    memory=memory
)

# Chat function
def chat(query):
    result = qa_chain({"question": query})
    return result['answer']

# Example usage
print(chat("Who is Komal Vardhan Lolugu?"))
# print(chat("How does vector search work?"))
# print(chat("What can I use LangChain for?"))
