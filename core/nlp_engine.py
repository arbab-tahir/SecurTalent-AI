import numpy as np
from sentence_transformers import SentenceTransformer
import os

class SemanticMappingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the Local CPU-Optimized Transformer Framework.
        Downloads weights (~80MB) on first boot, caching them locally.
        """
        # Suppress tokenization warnings across terminal outputs
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer weights: {str(e)}")

    def calculate_cosine_similarity(self, vector_a, vector_b):
        """
        Computes spatial angular distance using the Cosine Similarity proof:
        Similarity = (A . B) / (||A|| ||B||)
        """
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        # Exception Gate: Mitigate division-by-zero risk if an vector is completely empty
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))

    def evaluate_alignment(self, document_text, requirement_profile):
        """
        Transforms text profiles into discrete multi-dimensional vector arrays 
        and extracts similarity indices.
        """
        # Handle structural edge case if inputs are completely blank strings
        if not document_text.strip() or not requirement_profile.strip():
            return 0.0

        # Structural Vectorization Injection Layer
        embeddings = self.model.encode([document_text, requirement_profile])
        
        doc_vector = embeddings[0]
        req_vector = embeddings[1]
        
        # Calculate mathematical similarity index
        similarity_score = self.calculate_cosine_similarity(doc_vector, req_vector)
        return round(similarity_score, 4)
