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
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the dashboard
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
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SecurTalent-AI")
st.subheader("Unified Recruiter Evaluation Console & Semantic Spoofing Defense")

# --- Resource Initialization ---
@st.cache_resource(show_spinner="Initializing Neural Architecture (all-MiniLM-L6-v2)...")
def load_nlp_engine():
    return SemanticMappingEngine()

nlp_engine = load_nlp_engine()
provenance_engine = ProvenanceEngine()
fusion_engine = FusionEngine()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    st.markdown("Adjust mathematical fusion weights:")
    w1 = st.slider("Semantic Alignment Weight ($w_1$)", 0.0, 1.0, 0.6, 0.1)
    w2 = st.slider("Provenance Confidence Weight ($w_2$)", 0.0, 1.0, 0.4, 0.1)
    
    st.markdown("---")
    st.info("The Multi-Layer Fusion Engine dynamically balances semantic proximity with verified public identity records.")
    
    # Update Fusion Engine with dynamic weights
    fusion_engine = FusionEngine(weight_similarity=w1, weight_provenance=w2)

# --- Main Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📄 1. Document Intake Stream")
    uploaded_file = st.file_uploader("Upload Untrusted Candidate PDF", type=["pdf"])
    
with col2:
    st.markdown("### 🎯 2. Requirement Matrix")
    job_requirements = st.text_area(
        "Enter Job Requirements / Semantic Baseline", 
        height=150, 
        placeholder="e.g. Senior Python developer with strong skills in Machine Learning, PyTorch, and deploying REST APIs..."
    )

if uploaded_file is not None and job_requirements:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Execute Deterministic Defense Pipeline", use_container_width=True, type="primary"):
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
                st.markdown("### 🔍 Security Audit Results")
                
                # Note: pdfplumber warnings might show in terminal, this is normal
                integrity_results = audit_document_integrity(tmp_path)
                
                is_compromised = integrity_results["is_compromised"]
                cleaned_text = integrity_results["cleaned_text"]
                
                if is_compromised:
                    st.markdown(f"""
                    <div class="svwa-alert">
                        <h3>🚨 Sub-Visual Whitelisting Attack (SVWA) Detected!</h3>
                        <p><strong>Anomaly Ratio:</strong> {integrity_results['anomaly_ratio']:.2%}</p>
                        <p>The candidate attempted to exploit the parser using zero-contrast hidden text.</p>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Adversarial Glyphs (Raw Hex Data)"):
                        st.json(integrity_results["adversarial_payloads"][:10]) # Show top 10
                else:
                    st.success("✅ **Layer 1 [Integrity Guard] Passed:** No sub-visual primitive manipulation detected.")
                
                # ==========================================
                # LAYER 2: Semantic NLP
                # ==========================================
                similarity_score = nlp_engine.evaluate_alignment(cleaned_text, job_requirements)
                
                # ==========================================
                # LAYER 3: Provenance Audit
                # ==========================================
                provenance_score = provenance_engine.evaluate_provenance(cleaned_text)
                
                # ==========================================
                # LAYER 4: Fusion Engine
                # ==========================================
                final_score = fusion_engine.calculate_final_score(similarity_score, provenance_score, is_compromised)
                
                # ==========================================
                # VISUALIZATION
                # ==========================================
                st.markdown("---")
                st.markdown("### 📊 Empirical Output Matrix")
                
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
                        help="Confidence score based on GitHub REST API historical record parsing."
                    )
                with m3:
                    st.metric(
                        label="L4: Authenticity Coefficient ($C_{final}$)", 
                        value=f"{final_score:.4f}",
                        delta="-80.00% (SVWA Penalty)" if is_compromised else None,
                        delta_color="inverse"
                    )
                
                if is_compromised:
                    st.warning("⚠️ **System Intervention:** The multi-layer mathematical fusion engine multiplicatively applied a severe dynamic adversarial penalty factor ($P_{adv} = 0.20$) to the final index due to the Layer 1 security violation.")
                    
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
