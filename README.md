# SecurTalent-AI: A Deterministic Framework for Mitigating Adversarial Sub-Visual Injections and Semantic Spoofing in Automated Document Pipelines 🛡️

SecurTalent AI is an enterprise-grade, security-aware talent intelligence framework engineered to treat unstructured document intake streams as unverified, potentially adversarial data vectors. 

While legacy automated parsing applications blindly accept raw file data—leaving them vulnerable to manipulation—SecurTalent AI implements a deterministic multi-layer defense pipeline. The platform combines low-level document metadata auditing to catch sub-visual text manipulation with unsupervised deep learning vector space mapping to achieve contextual semantic alignment completely independent of literal keyword matches.

---

## 🏗️ System Architecture & Data Pipeline

The framework processes incoming binary document files through three decoupled verification layers before fusing the metrics into an empirical output matrix:

```text
[ Input: Untrusted Resume PDF ] ──► [ Input: Job Requirement Matrix ]
               │                                    │
               ▼                                    ▼
┌───────────────────────────────┐    ┌───────────────────────────────┐
│    Layer 1: Integrity Guard   │    │     Layer 2: Semantic NLP     │
│ (Sub-Visual Manipulation Check)    │  (Bi-Encoder Vector Mapping)  │
└───────────────┬───────────────┘    └───────────────┬───────────────┘
                │                                    │
                ▼                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│               Layer 3: Asynchronous Identity Attestation           │
│         (Pings Public REST APIs ──► Core Language Verification)    │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                Multi-Layer Mathematical Fusion Engine              │
│             (Calculates Authenticity Coefficient: Ca)              │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
         [ Streamlit Unified Recruiter Evaluation Console ]
```

---

## 🧱 Core Engineering Modules

### 1. Deterministic Contrast Ratio Auditor (`integrity_guard.py`)
Intercepts raw PDF character primitives and fonts before flattening text layers. Using `pdfplumber`, the system extracts every glyph's `non_stroking_color` matrix and applies the **W3C Relative Luminance Formula**:

$$\mathcal{L} = 0.2126R + 0.7152G + 0.0722B$$

The framework then computes the mathematical contrast ratio ($\mathcal{CR}$) between the extracted text glyph ($\mathcal{L}_1$) and the underlying document background canvas ($\mathcal{L}_2$) using the deterministic boundary rule:

$$\mathcal{CR} = \frac{\mathcal{L}_1 + 0.05}{\mathcal{L}_2 + 0.05}$$

If the computed ratio falls below the critical threshold ($\mathcal{CR} < 1.1$) within highly concentrated keyword zones, the system isolates the text coordinate and flags a **Sub-Visual Whitelisting Attack (SVWA)**.

### 2. Unsupervised Semantic Vector Space Mapper (`nlp_engine.py`)
Eliminates primitive, easily manipulated keyword-counting rules. Text inputs are processed through a pre-trained bi-encoder neural network framework (`all-MiniLM-L6-v2`), transforming unstructured text strings into dense 384-dimensional spatial coordinate vectors. The system evaluates the deep contextual proximity between the document vector ($A$) and the requirement vector ($B$) utilizing **Angular Cosine Similarity Matrix Math**:

$$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

This mathematical mapping ensures that conceptual equivalents are matched accurately, regardless of literal spelling variations.

### 3. Asynchronous Identity & Provenance Engine (`provenance_audit.py`)
Programmatically cross-references asserted resume claims with live external records. The engine deploys regular expression matching networks to safely isolate public platform handles and triggers non-blocking asynchronous REST requests to fetch public user metadata repositories. It calculates an **Authenticity Confidence Coefficient ($C_a$)** to score profile inflation risks based on historical code recency and language distribution matches.

### 4. Multi-Layer Mathematical Fusion Engine (`pipeline/fusion_engine.py`)
The system synthesizes the metrics from the defensive, semantic, and identity modules into a unified evaluation matrix by calculating an empirical **Authenticity Coefficient ($C_a$)**:

$$C_a = w_1(S_{\text{similarity}}) + w_2(G_{\text{provenance}})$$

Where $w_1$ and $w_2$ represent assigned operational weight parameters balancing semantic alignment and verified code provenance records, subjected to the constraint:

$$\sum_{i} w_i = 1.0$$

If Layer 1 triggers a security violation ($CR < 1.1$), a severe dynamic adversarial penalty factor ($P_{adv} = 0.20$) is applied multiplicatively to the model:

$$C_{\text{final}} = C_a \times P_{adv}$$

---

## 💻 Technical Stack & Dependencies

- **Runtime Backplane:** Python 3.10+
- **Document Rendering Object Inspector:** `pdfplumber`
- **Neural Network Transformers:** `sentence-transformers` (Local CPU-Optimized)
- **Mathematical Analysis Packages:** `NumPy` & `pandas`
- **Asynchronous Network Interface:** `requests` & `urllib3`
- **Researcher Console Layer:** `Streamlit`

---

## 📂 Repository Directory Matrix

```text
SecurTalent-AI/
├── core/
│   ├── __init__.py
│   ├── integrity_guard.py   # Layer 1: Contrast ratio & glyph metadata audits
│   ├── nlp_engine.py        # Layer 2: Sentence Transformers & vector similarity loops
│   └── provenance_audit.py  # Layer 3: Live public REST API connection matrix
├── pipeline/
│   ├── __init__.py
│   └── fusion_engine.py     # Aggregates analytical metrics to compute Ca index
├── interface/
│   └── dashboard.py         # Streamlit visual metrics display pipeline
├── requirements.txt         # Verified deployment configuration dependencies
└── README.md                # Framework documentation & technical summary
```

---

## 🚀 Local Deployment & Verification

1. Clone the core research repository infrastructure:
   ```bash
   git clone https://github.com/arbab-tahir/SecurTalent-AI.git
   cd SecurTalent-AI
   ```
2. Provision dependencies inside a localized virtual sandbox:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the visual evaluation console:
   ```bash
   streamlit run interface/dashboard.py
   ```
