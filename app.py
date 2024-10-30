import openai
import gradio as gr
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from datasets import load_dataset
import faiss
import numpy as np
import os

# Set your OpenAI API key
openai.api_key = os.getenv("testkey") 

# Initialize global variables for models and datasets
dataset = None
index = None
sbert_model = None
bm25 = None

# Step 1: Load dataset and create FAISS index and BM25 index
def load_data_and_create_index():
    global dataset, index, sbert_model, bm25
    # Load a smaller subset of the dataset to speed up testing (e.g., 10% of the data)
    dataset = load_dataset('ms_marco', 'v2.1', split='train[:5000]')  # Example: smaller set for testing

    # Load SBERT model for high-quality sentence embeddings
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight model for efficient encoding
    
    # Extract the first passage from the dataset (adjusting for its structure)
    data_contexts = [item['passages']['passage_text'][0] for item in dataset]  # Adjust for nested structure
    
    # Build BM25 index
    tokenized_corpus = [doc.split(" ") for doc in data_contexts]
    bm25 = BM25Okapi(tokenized_corpus)  # Initialize BM25 index
    
    # Encode contexts with SBERT to create embeddings
    data_vectors = sbert_model.encode(data_contexts, convert_to_tensor=True)
    
    # Initialize FAISS index for vector search
    dimension = data_vectors.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance metric
    index.add(np.array(data_vectors, dtype='float32'))  # Add encoded vectors to the FAISS index

# Step 2: Perform BM25 retrieval
def bm25_retrieval(query, top_k=5):
    # Tokenize query and perform BM25 search
    query_tokens = query.split(" ")
    results = bm25.get_top_n(query_tokens, [item['passages']['passage_text'][0] for item in dataset], n=top_k)
    return results

# Step 3: Retrieve Neighbors using FAISS (SBERT Embeddings)
def retrieve_neighbors(query, sbert_model, faiss_index, dataset, num_neighbors=5):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    distances, indices = faiss_index.search(np.array([query_embedding], dtype='float32'), num_neighbors)
    neighbors = [dataset[int(idx)]['passages']['passage_text'][0] for idx in indices[0]]  # Adjust for structure
    return neighbors

# Step 4: Generate Response using GPT-4 and Context
def generate_response_gpt4(query, context_snippets, max_tokens=850, temperature=0.7):
    context = " ".join(context_snippets[:10])  # Use top 10 neighbors as context
    input_text = f"Q: {query}\nHere is some relevant information:\n{context}\nBased on this information, provide a unique answer to the question."

    try:
        # Use the new ChatCompletion method from OpenAI v1.0.0+
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Specify GPT-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=max_tokens  # Adjust based on the expected response length
            ,temperature =temperature
            # Set clean_up_tokenization_spaces to True to suppress warning
             
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Step 5: Handle the query (integrate BM25 and FAISS retrieval)
def handle_query(query, num_neighbors, max_tokens, temperature):
    bm25_contexts = bm25_retrieval(query, top_k=num_neighbors)
    faiss_neighbors = retrieve_neighbors(query, sbert_model, index, dataset, num_neighbors)
    combined_contexts = bm25_contexts + faiss_neighbors
    response = generate_response_gpt4(query, combined_contexts, max_tokens=max_tokens, temperature=temperature)
    return "\n\n".join(combined_contexts), response

# Step 6: Define Gradio Interface
def interface(query, num_neighbors, max_tokens, temperature):
    neighbors, response = handle_query(query, num_neighbors, max_tokens, temperature)
    return neighbors, response

# Step 7: Load data, create FAISS and BM25 index, and start the interface
load_data_and_create_index()

# Create Gradio interface with sliders for neighbors, max tokens, and temperature
gr.Interface(
    fn=interface, 
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your query here..."), 
        gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Number of Neighbors to Retrieve"),
        gr.Slider(minimum=50, maximum=1500, step=50, value=850, label="Max Tokens"),  # Added max_tokens slider
        gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="Temperature")     # Added temperature slider
    ], 
    outputs=[
        gr.Textbox(label="Retrieved Context (BM25 and FAISS)"), 
        gr.Textbox(label="AI Generated Response")
    ],
    title="AI Query System with Contextual Data Retrieval (BM25 + SBERT) and GPT-4",
    description="Submit a query. The system retrieves relevant passages using BM25 and FAISS, then GPT-4 generates a response with adjustable max tokens and temperature."
).launch()
