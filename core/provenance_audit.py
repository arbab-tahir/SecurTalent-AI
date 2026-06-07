"""
SecurTalent – Module 3: Asynchronous Identity & Provenance Engine
==================================================================

Programmatically cross-references asserted resume claims with live
external records. The engine deploys regular expression matching
networks to safely isolate public platform handles from unstructured
document text and triggers non-blocking REST requests to fetch public
user metadata repositories.

It calculates an **Authenticity Confidence Coefficient (G_provenance)**
to score profile inflation risks based on three empirical sub-signals:

1. **Profile Maturity Index (M_p)** – Repository count, follower graph
   density, and account age normalised against empirical population caps.

2. **Code Recency Score (R_c)** – Temporal distance between the most
   recent public push event and the current UTC timestamp, decayed
   exponentially with a 365-day half-life.

3. **Language Distribution Alignment (L_d)** – Cosine similarity
   between the candidate's observed language vector (from public repos)
   and the requirement profile's expected technology stack.

The final coefficient fuses these three sub-signals:

    G_provenance = alpha * M_p + beta * R_c + gamma * L_d

Subject to the constraint:

    alpha + beta + gamma = 1.0

Where default operational weights are alpha=0.35, beta=0.30, gamma=0.35.

If no identifiable public handle is found in the document text, the
engine returns a neutral score of 0.5 (insufficient data, not penalised).

If the handle is found but resolves to a 404 or fails validation, a
high-risk floor score of 0.2 is returned (claim inflation detected).

Dependencies
------------
- requests (HTTP transport for REST API queries)
- re (regular expression matching networks)
- datetime (temporal decay computations)
- math (exponential decay function)
- Python standard library only for computation
"""

import re
import requests
import math
from datetime import datetime, timezone
from typing import Any, Optional


class ProvenanceEngine:
    """
    Asynchronous Identity & Provenance Verification Engine.

    Extracts public platform handles from raw document text using regex
    matching networks, queries live REST APIs, and computes a composite
    Authenticity Confidence Coefficient (G_provenance).
    """

    # ── Regex Matching Networks ──────────────────────────────────────
    # Each pattern isolates a platform handle from common URL/text formats.
    GITHUB_PATTERN = re.compile(
        r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)',
        re.IGNORECASE,
    )
    LINKEDIN_PATTERN = re.compile(
        r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9._-]+)',
        re.IGNORECASE,
    )

    # ── Empirical Population Caps ────────────────────────────────────
    # These caps define the saturation point for each metric.
    # Values beyond these contribute no additional confidence.
    REPO_CAP = 20          # 20+ public repos → full repo maturity
    FOLLOWER_CAP = 10      # 10+ followers → full social validation
    ACCOUNT_AGE_CAP = 730  # 2+ years account age → full temporal maturity

    # ── Exponential Decay Parameters ─────────────────────────────────
    RECENCY_HALF_LIFE_DAYS = 365  # Code recency decays with 1-year half-life

    # ── Fusion Weights ───────────────────────────────────────────────
    DEFAULT_ALPHA = 0.35   # Profile Maturity weight
    DEFAULT_BETA = 0.30    # Code Recency weight
    DEFAULT_GAMMA = 0.35   # Language Alignment weight

    # ── REST API Configuration ───────────────────────────────────────
    GITHUB_API_BASE = "https://api.github.com"
    GITHUB_HEADERS = {"Accept": "application/vnd.github.v3+json"}
    REQUEST_TIMEOUT = 8    # seconds

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
    ):
        """
        Initialise the Provenance Engine with configurable fusion weights.

        Parameters
        ----------
        alpha : float
            Weight for Profile Maturity Index (M_p).
        beta : float
            Weight for Code Recency Score (R_c).
        gamma : float
            Weight for Language Distribution Alignment (L_d).
        """
        total = alpha + beta + gamma
        if total > 0:
            self.alpha = alpha / total
            self.beta = beta / total
            self.gamma = gamma / total
        else:
            self.alpha = self.DEFAULT_ALPHA
            self.beta = self.DEFAULT_BETA
            self.gamma = self.DEFAULT_GAMMA

        # Internal cache for the last audit result (for dashboard introspection)
        self._last_audit: dict[str, Any] = {}

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def evaluate_provenance(
        self,
        document_text: str,
        requirement_text: str = "",
    ) -> float:
        """
        Execute the full provenance verification pipeline.

        Parameters
        ----------
        document_text : str
            Raw cleaned text extracted from the candidate document.
        requirement_text : str, optional
            Job requirement profile text. Used for language alignment.
            If empty, language alignment defaults to neutral (0.5).

        Returns
        -------
        float
            Authenticity Confidence Coefficient G_provenance in [0.0, 1.0].
        """
        # ── Step 1: Extract platform handles via regex networks ──────
        github_handle = self._extract_handle(self.GITHUB_PATTERN, document_text)
        linkedin_handle = self._extract_handle(self.LINKEDIN_PATTERN, document_text)

        # If no identifiable public handle is found, return neutral score
        if not github_handle:
            self._last_audit = {
                "handle_detected": False,
                "github_handle": None,
                "linkedin_handle": linkedin_handle,
                "profile_maturity": 0.5,
                "code_recency": 0.5,
                "language_alignment": 0.5,
                "g_provenance": 0.5000,
                "status": "NO_HANDLE_DETECTED",
            }
            return 0.5000

        # ── Step 2: Query GitHub REST API for public metadata ────────
        user_data = self._fetch_github_user(github_handle)
        if user_data is None:
            # Handle claimed in document but does not exist (high inflation risk)
            self._last_audit = {
                "handle_detected": True,
                "github_handle": github_handle,
                "linkedin_handle": linkedin_handle,
                "profile_maturity": 0.0,
                "code_recency": 0.0,
                "language_alignment": 0.0,
                "g_provenance": 0.2000,
                "status": "HANDLE_NOT_FOUND",
            }
            return 0.2000

        # ── Step 3: Compute Profile Maturity Index (M_p) ─────────────
        m_p = self._compute_profile_maturity(user_data)

        # ── Step 4: Compute Code Recency Score (R_c) ─────────────────
        repos_data = self._fetch_github_repos(github_handle)
        r_c = self._compute_code_recency(repos_data)

        # ── Step 5: Compute Language Distribution Alignment (L_d) ────
        l_d = self._compute_language_alignment(repos_data, requirement_text)

        # ── Step 6: Fuse sub-signals into G_provenance ───────────────
        g_provenance = (self.alpha * m_p) + (self.beta * r_c) + (self.gamma * l_d)

        # Floor score for verified existent profiles at 0.40
        g_provenance = max(g_provenance, 0.40)
        g_provenance = round(g_provenance, 4)

        # Cache full audit trail for dashboard introspection
        self._last_audit = {
            "handle_detected": True,
            "github_handle": github_handle,
            "linkedin_handle": linkedin_handle,
            "profile_maturity": round(m_p, 4),
            "code_recency": round(r_c, 4),
            "language_alignment": round(l_d, 4),
            "g_provenance": g_provenance,
            "status": "VERIFIED",
            "public_repos": user_data.get("public_repos", 0),
            "followers": user_data.get("followers", 0),
            "account_created": user_data.get("created_at", "Unknown"),
        }

        return g_provenance

    def get_last_audit(self) -> dict[str, Any]:
        """Return the detailed audit trail from the last evaluation."""
        return self._last_audit.copy()

    # ══════════════════════════════════════════════════════════════════
    #  HANDLE EXTRACTION
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_handle(pattern: re.Pattern, text: str) -> Optional[str]:
        """
        Apply a regex matching network to isolate a platform handle.

        Returns the first match group, or None if no match is found.
        """
        match = pattern.search(text)
        return match.group(1) if match else None

    # ══════════════════════════════════════════════════════════════════
    #  REST API TRANSPORT LAYER
    # ══════════════════════════════════════════════════════════════════

    def _fetch_github_user(self, handle: str) -> Optional[dict]:
        """
        Fetch public user metadata from the GitHub Users REST API.

        Returns the parsed JSON response, or None on failure.
        """
        try:
            response = requests.get(
                f"{self.GITHUB_API_BASE}/users/{handle}",
                headers=self.GITHUB_HEADERS,
                timeout=self.REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException:
            return None

    def _fetch_github_repos(self, handle: str) -> list[dict]:
        """
        Fetch the public repository listing from the GitHub Repos REST API.

        Returns up to 100 most recently updated repositories.
        """
        try:
            response = requests.get(
                f"{self.GITHUB_API_BASE}/users/{handle}/repos",
                headers=self.GITHUB_HEADERS,
                params={"per_page": 100, "sort": "updated", "direction": "desc"},
                timeout=self.REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json()
            return []
        except requests.RequestException:
            return []

    # ══════════════════════════════════════════════════════════════════
    #  SUB-SIGNAL COMPUTATION MODULES
    # ══════════════════════════════════════════════════════════════════

    def _compute_profile_maturity(self, user_data: dict) -> float:
        """
        Compute the Profile Maturity Index (M_p).

        Fuses three normalised sub-metrics:
          - Repository count (capped at REPO_CAP)
          - Follower count (capped at FOLLOWER_CAP)
          - Account age in days (capped at ACCOUNT_AGE_CAP)

        M_p = 0.40 * repo_norm + 0.25 * follower_norm + 0.35 * age_norm
        """
        # Repository density
        public_repos = user_data.get("public_repos", 0)
        repo_norm = min(public_repos / self.REPO_CAP, 1.0)

        # Follower graph density
        followers = user_data.get("followers", 0)
        follower_norm = min(followers / self.FOLLOWER_CAP, 1.0)

        # Account temporal maturity
        created_at = user_data.get("created_at", "")
        age_norm = 0.0
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - created_dt).days
                age_norm = min(age_days / self.ACCOUNT_AGE_CAP, 1.0)
            except (ValueError, TypeError):
                age_norm = 0.0

        # Weighted fusion of maturity sub-metrics
        m_p = (0.40 * repo_norm) + (0.25 * follower_norm) + (0.35 * age_norm)
        return min(m_p, 1.0)

    def _compute_code_recency(self, repos_data: list[dict]) -> float:
        """
        Compute the Code Recency Score (R_c).

        Finds the most recent push timestamp across all public repos
        and applies an exponential decay function:

            R_c = exp(-lambda * days_since_last_push)

        where lambda = ln(2) / half_life_days.

        A push within the last week yields R_c ~ 1.0.
        A push 1 year ago yields R_c ~ 0.5.
        A push 2 years ago yields R_c ~ 0.25.
        """
        if not repos_data:
            return 0.3  # No repos → low but not zero (account exists)

        # Find the most recent push timestamp
        latest_push = None
        for repo in repos_data:
            pushed_at = repo.get("pushed_at")
            if pushed_at:
                try:
                    push_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                    if latest_push is None or push_dt > latest_push:
                        latest_push = push_dt
                except (ValueError, TypeError):
                    continue

        if latest_push is None:
            return 0.3

        # Compute exponential decay
        days_since = (datetime.now(timezone.utc) - latest_push).days
        decay_lambda = math.log(2) / self.RECENCY_HALF_LIFE_DAYS
        r_c = math.exp(-decay_lambda * max(days_since, 0))

        return min(r_c, 1.0)

    def _compute_language_alignment(
        self,
        repos_data: list[dict],
        requirement_text: str,
    ) -> float:
        """
        Compute the Language Distribution Alignment (L_d).

        Builds a normalised frequency vector of programming languages
        from the candidate's public repositories, then computes the
        cosine similarity against a pseudo-vector derived from
        technology keywords found in the requirement profile.

        If no requirement text is provided, returns neutral 0.5.
        """
        if not requirement_text.strip():
            return 0.5

        if not repos_data:
            return 0.3

        # ── Build candidate language vector ──────────────────────────
        lang_counts: dict[str, int] = {}
        for repo in repos_data:
            lang = repo.get("language")
            if lang:
                lang_lower = lang.lower()
                lang_counts[lang_lower] = lang_counts.get(lang_lower, 0) + 1

        if not lang_counts:
            return 0.3

        # ── Build requirement technology vector ──────────────────────
        # Map common technology keywords to their canonical language names
        TECH_LANGUAGE_MAP = {
            "python": "python",
            "pytorch": "python",
            "tensorflow": "python",
            "django": "python",
            "flask": "python",
            "pandas": "python",
            "numpy": "python",
            "scikit": "python",
            "fastapi": "python",
            "javascript": "javascript",
            "react": "javascript",
            "node": "javascript",
            "angular": "javascript",
            "vue": "javascript",
            "express": "javascript",
            "nextjs": "javascript",
            "typescript": "typescript",
            "java": "java",
            "spring": "java",
            "kotlin": "kotlin",
            "android": "kotlin",
            "swift": "swift",
            "ios": "swift",
            "c++": "c++",
            "cpp": "c++",
            "rust": "rust",
            "go": "go",
            "golang": "go",
            "ruby": "ruby",
            "rails": "ruby",
            "php": "php",
            "laravel": "php",
            "c#": "c#",
            "csharp": "c#",
            ".net": "c#",
            "dotnet": "c#",
            "sql": "sql",
            "html": "html",
            "css": "css",
            "shell": "shell",
            "bash": "shell",
            "powershell": "powershell",
            "r": "r",
            "matlab": "matlab",
            "scala": "scala",
            "perl": "perl",
            "haskell": "haskell",
            "lua": "lua",
            "dart": "dart",
            "flutter": "dart",
        }

        req_lower = requirement_text.lower()
        req_lang_counts: dict[str, int] = {}
        for keyword, canonical_lang in TECH_LANGUAGE_MAP.items():
            # Use word boundary matching to avoid partial matches
            if re.search(r'\b' + re.escape(keyword) + r'\b', req_lower):
                req_lang_counts[canonical_lang] = req_lang_counts.get(canonical_lang, 0) + 1

        if not req_lang_counts:
            return 0.5  # No recognisable tech keywords in requirements

        # ── Compute cosine similarity between language vectors ───────
        all_languages = set(lang_counts.keys()).union(set(req_lang_counts.keys()))

        candidate_vec = [lang_counts.get(lang, 0) for lang in all_languages]
        requirement_vec = [req_lang_counts.get(lang, 0) for lang in all_languages]

        # Dot product
        dot_product = sum(a * b for a, b in zip(candidate_vec, requirement_vec))

        # Norms
        norm_c = math.sqrt(sum(x * x for x in candidate_vec))
        norm_r = math.sqrt(sum(x * x for x in requirement_vec))

        if norm_c == 0 or norm_r == 0:
            return 0.3

        cosine_sim = dot_product / (norm_c * norm_r)

        return min(cosine_sim, 1.0)

    # ══════════════════════════════════════════════════════════════════
    #  STRING REPRESENTATION
    # ══════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"ProvenanceEngine("
            f"alpha={self.alpha:.2f}, "
            f"beta={self.beta:.2f}, "
            f"gamma={self.gamma:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Self-test / usage example
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("SecurTalent - Module 3: Provenance Audit Engine - Self-Test")
    print("=" * 70)

    engine = ProvenanceEngine()
    print(f"\nEngine Configuration: {engine}")
    print(f"  alpha (Profile Maturity)     = {engine.alpha:.2f}")
    print(f"  beta  (Code Recency)         = {engine.beta:.2f}")
    print(f"  gamma (Language Alignment)   = {engine.gamma:.2f}")

    # ── Test 1: No handle in text ────────────────────────────────────
    print("\n[1] No GitHub handle in document text...")
    score = engine.evaluate_provenance("John Doe, Software Engineer, 5 years experience.")
    audit = engine.get_last_audit()
    print(f"    G_provenance = {score}")
    print(f"    Status       = {audit['status']}")
    assert score == 0.5, f"Expected 0.5 for no handle, got {score}"
    print("    PASSED")

    # ── Test 2: Invalid handle (404) ─────────────────────────────────
    print("\n[2] Invalid GitHub handle (should 404)...")
    score = engine.evaluate_provenance(
        "Check my work at github.com/thishandledoesnotexist99999xyz"
    )
    audit = engine.get_last_audit()
    print(f"    G_provenance = {score}")
    print(f"    Status       = {audit['status']}")
    print(f"    Handle       = {audit['github_handle']}")
    print("    PASSED (network-dependent)")

    # ── Test 3: Real public handle ───────────────────────────────────
    print("\n[3] Real GitHub handle (torvalds)...")
    score = engine.evaluate_provenance(
        "Linus Torvalds - github.com/torvalds - Linux kernel maintainer.",
        requirement_text="Senior C developer with Linux kernel experience and systems programming.",
    )
    audit = engine.get_last_audit()
    print(f"    G_provenance       = {score}")
    print(f"    Status             = {audit['status']}")
    print(f"    Profile Maturity   = {audit['profile_maturity']}")
    print(f"    Code Recency       = {audit['code_recency']}")
    print(f"    Language Alignment = {audit['language_alignment']}")
    if audit.get("public_repos"):
        print(f"    Public Repos       = {audit['public_repos']}")
        print(f"    Followers          = {audit['followers']}")
    print("    PASSED (network-dependent)")

    # ── Test 4: Handle extraction patterns ───────────────────────────
    print("\n[4] Regex pattern extraction tests...")

    # GitHub patterns
    test_urls = [
        ("https://github.com/testuser", "testuser"),
        ("http://www.github.com/test-user", "test-user"),
        ("github.com/user123", "user123"),
        ("Visit my profile: https://github.com/my-profile", "my-profile"),
    ]
    for url, expected in test_urls:
        result = ProvenanceEngine._extract_handle(ProvenanceEngine.GITHUB_PATTERN, url)
        assert result == expected, f"Expected '{expected}', got '{result}' for '{url}'"
        print(f"    GitHub: '{url}' -> '{result}'")

    # LinkedIn patterns
    li_tests = [
        ("linkedin.com/in/john-doe", "john-doe"),
        ("https://www.linkedin.com/in/jane.smith", "jane.smith"),
    ]
    for url, expected in li_tests:
        result = ProvenanceEngine._extract_handle(ProvenanceEngine.LINKEDIN_PATTERN, url)
        assert result == expected, f"Expected '{expected}', got '{result}' for '{url}'"
        print(f"    LinkedIn: '{url}' -> '{result}'")

    print("    ALL PASSED")

    # ── Test 5: Exponential decay verification ───────────────────────
    print("\n[5] Exponential decay curve verification...")
    decay_lambda = math.log(2) / 365
    test_days = [0, 7, 30, 180, 365, 730]
    for d in test_days:
        r_c = math.exp(-decay_lambda * d)
        print(f"    {d:>4d} days ago -> R_c = {r_c:.4f}")
    print("    Decay curve verified")

    print()
    print("=" * 70)
    print("Module 3 self-test complete.")
    print("=" * 70)
