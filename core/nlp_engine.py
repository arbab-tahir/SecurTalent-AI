import numpy as np
import os
import sys
import subprocess
from collections import Counter
import re

class SemanticMappingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the Local CPU-Optimized Transformer Framework.
        Falls back to a zero-dependency bag-of-words similarity engine
        if the neural weights or PyTorch runtime fails (e.g., segfaults/crashes on Windows).
        """
        # Suppress tokenization warnings across terminal outputs
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = None
        self.use_fallback = False
        
        # Check environment override
        if os.environ.get("USE_FALLBACK_NLP") == "1" or os.environ.get("USE_MOCK_NLP") == "1":
            print("[INFO] Fallback/Mock NLP engine forced via environment variable.")
            self.use_fallback = True
            return
            
        print("Checking Neural Engine integrity...")
        try:
            # Test loading the model in a subprocess to protect against hard segmentation faults
            test_code = (
                "import os; "
                "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'; "
                "os.environ['OMP_NUM_THREADS'] = '1'; "
                "from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{model_name}')"
            )
            cmd = [sys.executable, "-c", test_code]
            
            # Run with a short timeout to see if it succeeds
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                # Subprocess succeeded, meaning PyTorch and SentenceTransformer run fine on this CPU/OS
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                print("[SUCCESS] Neural Engine (all-MiniLM-L6-v2) successfully loaded.")
            else:
                # Subprocess failed (crashed, segfaulted, or threw python exception)
                self.use_fallback = True
                print(f"[WARNING] Neural Engine runtime check failed (Exit Code: {result.returncode}).")
                print("[SYSTEM] Activating zero-dependency mathematical fallback engine for stability.")
        except Exception as e:
            self.use_fallback = True
            print(f"[WARNING] Neural Engine check encountered an error: {e}")
            print("[SYSTEM] Activating zero-dependency mathematical fallback engine for stability.")

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

    def calculate_fallback_similarity(self, text_a, text_b):
        """
        Calculates similarity using a pure-Python TF-IDF / Cosine Similarity
        approach on word frequencies. This is extremely robust, has 0 dependencies,
        and dynamically responds to text content matches.
        """
        def get_tokens(text):
            # Clean and tokenize text
            words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
            # Very basic english stop words list to improve similarity accuracy
            stop_words = {
                'the', 'and', 'a', 'of', 'to', 'in', 'is', 'that', 'for', 'it', 'on', 'with', 
                'as', 'at', 'by', 'an', 'be', 'this', 'are', 'from', 'or', 'you', 'your', 'we'
            }
            return [w for w in words if w not in stop_words]
            
        tokens_a = get_tokens(text_a)
        tokens_b = get_tokens(text_b)
        
        if not tokens_a or not tokens_b:
            return 0.0
            
        counts_a = Counter(tokens_a)
        counts_b = Counter(tokens_b)
        
        # Unique vocabulary from both texts
        vocab = set(counts_a.keys()).union(set(counts_b.keys()))
        
        # Simple TF calculation
        vector_a = np.array([counts_a.get(word, 0) for word in vocab], dtype=float)
        vector_b = np.array([counts_b.get(word, 0) for word in vocab], dtype=float)
        
        # Calculate Cosine Similarity
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        similarity = dot_product / (norm_a * norm_b)
        
        # Soft scaling: Cosine similarity on word counts ranges lower than dense embeddings.
        # We take the square root to make the dynamic scale match typical embedding levels (e.g. 0.6 - 0.9).
        scaled_similarity = np.sqrt(similarity)
        
        return min(float(scaled_similarity), 1.0)

    def evaluate_alignment(self, document_text, requirement_profile):
        """
        Transforms text profiles into discrete multi-dimensional vector arrays 
        and extracts similarity indices.
        """
        # Handle structural edge case if inputs are completely blank strings
        if not document_text.strip() or not requirement_profile.strip():
            return 0.0

        if self.use_fallback:
            similarity_score = self.calculate_fallback_similarity(document_text, requirement_profile)
        else:
            try:
                # Structural Vectorization Injection Layer
                embeddings = self.model.encode([document_text, requirement_profile])
                doc_vector = embeddings[0]
                req_vector = embeddings[1]
                similarity_score = self.calculate_cosine_similarity(doc_vector, req_vector)
            except Exception as e:
                # Extra safety gate in case model.encode fails
                print(f"[ERROR] Embedding generation failed: {e}. Falling back...")
                similarity_score = self.calculate_fallback_similarity(document_text, requirement_profile)
        
        return round(similarity_score, 4)

