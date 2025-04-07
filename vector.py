pip install langchain-openai tiktoken python-dotenv langchain_community numpy scikit-learn langchain-community

import os
from langchain_openai import AzureOpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_API_ENBEDDINGS_DEPLOYMENT_NAME"),
)

# Sample sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn canine leaps above an idle hound.",
    "Artificial intelligence is revolutionizing technology.",
    "The weather is beautiful today."
]

# Vectorize the sentences
vectors = embeddings.embed_documents(sentences)

# Function to find the most similar sentence
def find_most_similar(query, vectors, sentences):
    query_vector = embeddings.embed_query(query)
    similarities = cosine_similarity([query_vector], vectors)[0]
    most_similar_idx = np.argmax(similarities)
    return sentences[most_similar_idx], similarities[most_similar_idx]

# Example usage
query = "Komal Vardhan is a Developer"
most_similar, similarity = find_most_similar(query, vectors, sentences)

print(f"Query: {query}")
print(f"Most similar sentence: {most_similar}")
print(f"Similarity score: {similarity}")
print(f"Similarity Percentage: {round(similarity*100,2)}%")
