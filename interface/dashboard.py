"""
SecurTalent – Streamlit Unified Recruiter Evaluation Console
=============================================================

Interactive analysis interface integrating all four detection layers:
  Layer 1: Integrity Guard (Sub-Visual Whitelisting Attack Detection)
  Layer 2: Semantic NLP Engine (Bi-Encoder Vector Space Mapping)
  Layer 3: Provenance Audit Engine (Live REST API Identity Verification)
  Layer 4: Multi-Layer Mathematical Fusion Engine (Risk Classification)
"""

import streamlit as st
import tempfile
import os
import sys

# Prevent hard crashes / connection errors on Windows with PyTorch + Streamlit
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure the root project directory is in the Python path
# This prevents "ModuleNotFoundError: No module named 'core'"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import core modules and pipeline
from core.integrity_guard import audit_document_integrity
from core.nlp_engine import SemanticMappingEngine
from core.provenance_audit import ProvenanceEngine
from pipeline.fusion_engine import FusionEngine

st.set_page_config(
    page_title="SecurTalent-AI Console",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .metric-container {
        background-color: #1E1E2E;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .svwa-alert {
        background-color: #2b080c;
        border-left: 5px solid #FF3333;
        padding: 15px;
        border-radius: 6px;
    }
    .svwa-alert h3 {
        color: #ff4d4d !important;
        margin-top: 0;
        font-family: monospace;
    }
    .svwa-alert p {
        color: #ffcccc !important;
        font-size: 1.1em;
    }
    .svwa-alert strong {
        color: #ffffff !important;
    }
    .risk-low {
        background-color: #0a2e0a;
        border-left: 5px solid #33cc33;
        padding: 15px;
        border-radius: 6px;
        color: #ccffcc !important;
    }
    .risk-moderate {
        background-color: #2e2a0a;
        border-left: 5px solid #ffcc00;
        padding: 15px;
        border-radius: 6px;
        color: #fff5cc !important;
    }
    .risk-high {
        background-color: #2e1a0a;
        border-left: 5px solid #ff8800;
        padding: 15px;
        border-radius: 6px;
        color: #ffe0cc !important;
    }
    .risk-critical {
        background-color: #2b080c;
        border-left: 5px solid #ff2222;
        padding: 15px;
        border-radius: 6px;
        color: #ffcccc !important;
    }
    .provenance-detail {
        background-color: #0f1a2e;
        border-left: 5px solid #4488ff;
        padding: 15px;
        border-radius: 6px;
        color: #cce0ff !important;
    }
    .provenance-detail strong {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("\U0001f6e1\ufe0f SecurTalent-AI")
st.subheader("Unified Recruiter Evaluation Console & Semantic Spoofing Defense")

# ══════════════════════════════════════════════════════════════════════
#  RESOURCE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initializing Neural Architecture (all-MiniLM-L6-v2)...")
def load_nlp_engine():
    return SemanticMappingEngine()

nlp_engine = load_nlp_engine()
provenance_engine = ProvenanceEngine()
fusion_engine = FusionEngine()

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### \u2699\ufe0f Engine Parameters")
    st.markdown("Adjust mathematical fusion weights:")
    w1 = st.slider("Semantic Alignment Weight ($w_1$)", 0.0, 1.0, 0.6, 0.1)
    w2 = st.slider("Provenance Confidence Weight ($w_2$)", 0.0, 1.0, 0.4, 0.1)

    st.markdown("---")
    st.markdown("### \U0001f9ea Provenance Engine Weights")
    p_alpha = st.slider("Profile Maturity ($\\alpha$)", 0.0, 1.0, 0.35, 0.05)
    p_beta = st.slider("Code Recency ($\\beta$)", 0.0, 1.0, 0.30, 0.05)
    p_gamma = st.slider("Language Alignment ($\\gamma$)", 0.0, 1.0, 0.35, 0.05)

    st.markdown("---")
    st.info("The Multi-Layer Fusion Engine dynamically balances semantic proximity with verified public identity records.")

    # Update engines with dynamic weights
    fusion_engine = FusionEngine(weight_similarity=w1, weight_provenance=w2)
    provenance_engine = ProvenanceEngine(alpha=p_alpha, beta=p_beta, gamma=p_gamma)

# ══════════════════════════════════════════════════════════════════════
#  MAIN INTERFACE
# ══════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### \U0001f4c4 1. Document Intake Stream")
    uploaded_file = st.file_uploader("Upload Untrusted Candidate PDF", type=["pdf"])

with col2:
    st.markdown("### \U0001f3af 2. Requirement Matrix")
    job_requirements = st.text_area(
        "Enter Job Requirements / Semantic Baseline",
        height=150,
        placeholder="e.g. Senior Python developer with strong skills in Machine Learning, PyTorch, and deploying REST APIs..."
    )

if uploaded_file is not None and job_requirements:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("\U0001f680 Execute Deterministic Defense Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Processing multi-layer verification pipeline..."):

            # Temporary storage for pdfplumber analysis
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # ==========================================
                # LAYER 1: Integrity Guard (SVWA Check)
                # ==========================================
                st.markdown("---")
                st.markdown("### \U0001f50d Layer 1: Integrity Guard")

                integrity_results = audit_document_integrity(tmp_path)

                is_compromised = integrity_results["is_compromised"]
                cleaned_text = integrity_results["cleaned_text"]

                if is_compromised:
                    st.markdown(f"""
                    <div class="svwa-alert">
                        <h3>\U0001f6a8 Sub-Visual Whitelisting Attack (SVWA) Detected!</h3>
                        <p><strong>Anomaly Ratio:</strong> {integrity_results['anomaly_ratio']:.2%}</p>
                        <p>The candidate attempted to exploit the parser using zero-contrast hidden text.</p>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

                    with st.expander("View Adversarial Glyphs (Raw Hex Data)"):
                        st.json(integrity_results["adversarial_payloads"][:10])
                else:
                    st.success("\u2705 **Layer 1 [Integrity Guard] Passed:** No sub-visual primitive manipulation detected.")

                # ==========================================
                # LAYER 2: Semantic NLP
                # ==========================================
                st.markdown("---")
                st.markdown("### \U0001f9e0 Layer 2: Semantic Vector Mapping")

                similarity_score = nlp_engine.evaluate_alignment(cleaned_text, job_requirements)

                if similarity_score >= 0.70:
                    st.success(f"\u2705 **Strong Semantic Alignment:** $S_{{similarity}}$ = {similarity_score:.4f}")
                elif similarity_score >= 0.45:
                    st.warning(f"\u26a0\ufe0f **Partial Semantic Alignment:** $S_{{similarity}}$ = {similarity_score:.4f}")
                else:
                    st.error(f"\u274c **Weak Semantic Alignment:** $S_{{similarity}}$ = {similarity_score:.4f}")

                # ==========================================
                # LAYER 3: Provenance Audit
                # ==========================================
                st.markdown("---")
                st.markdown("### \U0001f50e Layer 3: Identity & Provenance Verification")

                provenance_score = provenance_engine.evaluate_provenance(
                    cleaned_text,
                    requirement_text=job_requirements,
                )
                prov_audit = provenance_engine.get_last_audit()

                if prov_audit.get("status") == "NO_HANDLE_DETECTED":
                    st.info("\U0001f4ad **No public platform handle detected in document.** Provenance score defaulted to neutral (0.5000).")
                elif prov_audit.get("status") == "HANDLE_NOT_FOUND":
                    st.error(f"\U0001f6a8 **Profile Claim Inflation Detected:** GitHub handle `{prov_audit.get('github_handle')}` does not resolve. Score penalised to 0.2000.")
                elif prov_audit.get("status") == "VERIFIED":
                    detail_html = f"""
                    <div class="provenance-detail">
                        <strong>\u2705 GitHub Profile Verified:</strong> {prov_audit.get('github_handle', 'N/A')}<br>
                        <strong>Public Repos:</strong> {prov_audit.get('public_repos', 'N/A')} |
                        <strong>Followers:</strong> {prov_audit.get('followers', 'N/A')} |
                        <strong>Account Created:</strong> {prov_audit.get('account_created', 'N/A')}<br><br>
                        <strong>Profile Maturity ($M_p$):</strong> {prov_audit.get('profile_maturity', 0):.4f} |
                        <strong>Code Recency ($R_c$):</strong> {prov_audit.get('code_recency', 0):.4f} |
                        <strong>Language Alignment ($L_d$):</strong> {prov_audit.get('language_alignment', 0):.4f}
                    </div>
                    <br>
                    """
                    st.markdown(detail_html, unsafe_allow_html=True)

                # ==========================================
                # LAYER 4: Fusion Engine
                # ==========================================
                final_score = fusion_engine.calculate_final_score(similarity_score, provenance_score, is_compromised)
                fusion_result = fusion_engine.get_last_result()
                risk_tier = fusion_result.get("risk_tier", "CRITICAL")

                # ==========================================
                # VISUALIZATION: Empirical Output Matrix
                # ==========================================
                st.markdown("---")
                st.markdown("### \U0001f4ca Empirical Output Matrix")

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric(
                        label="L2: Semantic Alignment ($S_{similarity}$)",
                        value=f"{similarity_score:.4f}",
                        help="Cosine similarity mapped in 384-dimensional space."
                    )
                with m2:
                    st.metric(
                        label="L3: Identity Attestation ($G_{provenance}$)",
                        value=f"{provenance_score:.4f}",
                        help="Composite confidence score from GitHub REST API: profile maturity, code recency, and language alignment."
                    )
                with m3:
                    st.metric(
                        label="L4: Authenticity Coefficient ($C_{final}$)",
                        value=f"{final_score:.4f}",
                        delta="-80.00% (SVWA Penalty)" if is_compromised else None,
                        delta_color="inverse"
                    )

                # ==========================================
                # RISK CLASSIFICATION BANNER
                # ==========================================
                st.markdown("---")
                st.markdown("### \U0001f3af Risk Classification")

                risk_css_class = f"risk-{risk_tier.lower()}"
                risk_icon = {
                    "LOW": "\u2705",
                    "MODERATE": "\u26a0\ufe0f",
                    "HIGH": "\U0001f6a8",
                    "CRITICAL": "\u2620\ufe0f",
                }.get(risk_tier, "\u2753")

                risk_label = fusion_result.get("risk_label", "Unknown")

                st.markdown(f"""
                <div class="{risk_css_class}">
                    <h3 style="margin-top:0;">{risk_icon} {risk_tier} RISK</h3>
                    <p style="font-size:1.1em;">{risk_label}</p>
                    <p><strong>Base Coefficient ($C_a$):</strong> {fusion_result.get('c_a_base', 0):.4f} |
                    <strong>Penalty Factor:</strong> x{fusion_result.get('penalty_applied', 1.0)} |
                    <strong>Final ($C_{{final}}$):</strong> {final_score:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

                if is_compromised:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.warning("\u26a0\ufe0f **System Intervention:** The multi-layer mathematical fusion engine multiplicatively applied a severe dynamic adversarial penalty factor ($P_{adv} = 0.20$) to the final index due to the Layer 1 security violation.")

                # ==========================================
                # DETAILED BREAKDOWN (Expandable)
                # ==========================================
                st.markdown("---")
                with st.expander("View Full Diagnostic Breakdown"):
                    diag_col1, diag_col2 = st.columns(2)

                    with diag_col1:
                        st.markdown("**Fusion Engine Parameters**")
                        st.json({
                            "w1_semantic": fusion_result.get("w1", 0),
                            "w2_provenance": fusion_result.get("w2", 0),
                            "c_a_base": fusion_result.get("c_a_base", 0),
                            "penalty_factor": fusion_result.get("penalty_applied", 1.0),
                            "c_final": fusion_result.get("c_final", 0),
                            "risk_tier": risk_tier,
                        })

                    with diag_col2:
                        st.markdown("**Provenance Audit Trail**")
                        st.json(prov_audit)

            finally:
                # Cleanup temporary PDF
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
