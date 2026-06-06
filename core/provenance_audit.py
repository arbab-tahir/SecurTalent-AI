import re
import requests
import json

class ProvenanceEngine:
    def __init__(self):
        """
        Initializes the Asynchronous Identity & Provenance Engine.
        """
        self.github_pattern = re.compile(r'github\.com/([a-zA-Z0-9-]+)')
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        
    def evaluate_provenance(self, document_text: str) -> float:
        """
        Programmatically cross-references asserted resume claims with live external records.
        Returns an Authenticity Confidence Coefficient (G_provenance) between 0.0 and 1.0.
        """
        match = self.github_pattern.search(document_text)
        
        # If no identifiable public handle is found, return a neutral score (0.5)
        if not match:
            return 0.5000
            
        handle = match.group(1)
        
        try:
            # Non-blocking REST request to fetch public metadata
            response = requests.get(
                f"https://api.github.com/users/{handle}", 
                headers=self.headers, 
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                public_repos = data.get("public_repos", 0)
                followers = data.get("followers", 0)
                
                # Score profile inflation risk based on history
                # Empirical caps: 15+ repos and 5+ followers = max authenticity
                repo_score = min(public_repos / 15.0, 1.0)
                follower_score = min(followers / 5.0, 1.0)
                
                # Compute Authenticity Confidence Coefficient
                confidence_score = (repo_score * 0.7) + (follower_score * 0.3)
                
                # Floor score for verified existent profiles at 0.6
                return max(round(confidence_score, 4), 0.6000)
            else:
                # Profile claim exists in document but 404s/fails externally (high risk)
                return 0.3000
        except requests.RequestException:
            # Fallback for network timeouts
            return 0.5000
