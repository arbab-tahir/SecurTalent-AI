class FusionEngine:
    def __init__(self, weight_similarity: float = 0.6, weight_provenance: float = 0.4, adversarial_penalty: float = 0.20):
        """
        Initializes the Multi-Layer Mathematical Fusion Engine.
        """
        self.w1 = weight_similarity
        self.w2 = weight_provenance
        self.p_adv = adversarial_penalty
        
    def calculate_final_score(self, similarity: float, provenance: float, is_compromised: bool) -> float:
        """
        Synthesizes metrics from defensive, semantic, and identity modules.
        
        Ca = w1(S_similarity) + w2(G_provenance)
        If compromised: C_final = Ca * P_adv
        """
        # Ensure operational weight parameters sum to 1.0
        total_weight = self.w1 + self.w2
        if total_weight > 0:
            w1_norm = self.w1 / total_weight
            w2_norm = self.w2 / total_weight
        else:
            w1_norm, w2_norm = 0.5, 0.5
            
        # Calculate Authenticity Coefficient (Ca)
        c_a = (w1_norm * similarity) + (w2_norm * provenance)
        
        # Apply severe dynamic adversarial penalty factor if Layer 1 triggered a security violation
        if is_compromised:
            c_final = c_a * self.p_adv
        else:
            c_final = c_a
            
        return round(c_final, 4)
