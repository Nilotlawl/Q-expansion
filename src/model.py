# src/model.py

import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import nltk


nltk.download('punkt', quiet=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



spanbert_model_name = "SpanBERT/spanbert-base-cased"
spanbert_tokenizer = AutoTokenizer.from_pretrained(spanbert_model_name)
spanbert_model = AutoModel.from_pretrained(spanbert_model_name)
spanbert_model.to(device)

def get_spanbert_embedding(text):
    """
    Get an embedding for the input text using SpanBERT.
    We compute the mean of the token embeddings from the last hidden state.
    """
    inputs = spanbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = spanbert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().cpu().numpy()

def resolve_coreferences(dialogue_history):
    """
    Resolve coreferences in the latest dialogue message using SpanBERT embeddings.
    
    For each pronoun in the latest message, this function searches previous dialogue messages,
    computes their embeddings, and selects the one with the highest cosine similarity as the antecedent.
    Then, it replaces the pronoun with that candidate antecedent.
    """
    latest_query = dialogue_history[-1]
    pronouns = ['he', 'she', 'it', 'his', 'her', 'its', 'they', 'them', 'their']
    latest_embedding = get_spanbert_embedding(latest_query)
    resolved_query = latest_query

    for pronoun in pronouns:
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        if re.search(pattern, resolved_query, flags=re.IGNORECASE):
            best_candidate = None
            best_score = -1.0
            # Search previous dialogue messages for the best candidate antecedent
            for message in dialogue_history[:-1]:
                candidate_embedding = get_spanbert_embedding(message)
                # Use cosine similarity from sentence-transformers utility
                score = util.cos_sim(torch.tensor(latest_embedding), torch.tensor(candidate_embedding)).item()
                if score > best_score:
                    best_score = score
                    best_candidate = message.strip()
            if best_candidate:
                resolved_query = re.sub(pattern, best_candidate, resolved_query, flags=re.IGNORECASE)
    return resolved_query


# Transformer-Based Query Refinement (using google-t5/t5-base)

def refine_query(query):
    """
    Refine and expand the query using the google-t5/t5-base model.
    The model and input tensors are moved to the CUDA device if available.
    """
    model_name = "google-t5/t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    t5_model.to(device)
    
    # Prepare input with a task prefix (here "expand:" is used to indicate the task)
    input_text = "expand: " + query
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    outputs = t5_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    refined_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined_query


# Embedding-Based Topic Classification
def classify_topic(query):
    """
    Classify the query into a topic using an embedding-based vector search.
    Pre-defined topic descriptions are embedded using SentenceTransformer.
    """
    topics = {
        "Politics - India": "Queries related to Indian politics, government, and public figures",
        "Politics - UK": "Queries related to UK politics, government, and public figures",
        "Sports - Football": "Queries related to football news and matches",
        "General": "General queries not fitting into specific categories"
    }
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    topic_embeddings = {
        topic: embedder.encode(desc, convert_to_tensor=True)
        for topic, desc in topics.items()
    }
    
    best_topic = None
    best_score = -1.0
    for topic, emb in topic_embeddings.items():
        score = util.pytorch_cos_sim(query_embedding, emb).item()
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic
def get_answer(refined_query):
    """
    Lookup an answer for the refined query from a simple dictionary.
    This simulates a chatbot that doesn't use a full LLM for generating replies.
    """
    # A simple mapping from refined queries to answers. In a real system, this might query a database or API.
    knowledge_base = {
        "Who is the Prime Minister of India?": "The Prime Minister of India is Narendra Modi.",
        "What are the Prime Minister of India's duties?": "The Prime Minister of India is responsible for leading the government, formulating policies, and representing the country both domestically and internationally."
    }
    
    # Return a matching answer if found, otherwise a default response.
    return knowledge_base.get(refined_query, "I'm sorry, I don't have an answer for that.")