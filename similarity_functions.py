# similarity_functions.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_normalized_similarity(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity_score = cosine_similarity([embeddings1[0]], [embeddings2[0]])[0][0]

    # Normalize and clip the score to be between 0 and 1
    normalized_score = (similarity_score + 1) / 2
    normalized_score = np.clip(normalized_score, 0, 1)
    return normalized_score