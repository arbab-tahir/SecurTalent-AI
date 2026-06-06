from core.nlp_engine import SemanticMappingEngine
import time

def run_phase_2_benchmark():
    print("==================================================================================")
    print("[INFO] SECURTALENT-AI: INITIALIZING SEMANTIC NLP MODEL VECTORIZATION BENCHMARK")
    print("==================================================================================")
    
    print("Loading Local Neural Architecture Model (all-MiniLM-L6-v2)...")
    start_time = time.time()
    engine = load_engine()
    print(f"Model successfully loaded and cached in {time.time() - start_time:.2f} seconds.\n")

    # [TARGET] REFERENCE BENCHMARK TARGET: Standard Telenor SOC/DFIR Job Requirement Profile
    telenor_job_profile = (
        "Seek an entry-level SOC Analyst proficient in threat detection, network monitoring, "
        "and incident response. Must have hands-on experience tracking alerts using SIEM tools "
        "like Wazuh or Splunk, parsing logs via Wireshark, capturing memory images with FTK Imager, "
        "and executing basic malware triage inside Linux environments."
    )

    # [TEST BATCH] PROFILE TEST BATCH: Three distinct conceptual resume content profiles
    test_resumes = {
        "Profile A (Perfect Synonyms - High Context Match)": (
            "Cybersecurity graduate specializing in DFIR. Skilled in digital evidence collection, "
            "packet capture analysis, and system log auditing. Experienced in deploying enterprise "
            "intrusion detection systems and network monitoring configurations to accelerate incident "
            "containment workflows."
        ),
        "Profile B (Partial / Keyword Stuffed Profile)": (
            "I want a job. I know basic IT systems and Windows. I am looking for an opportunity "
            "to grow my career. Python data science. Wireshark wazuh splunk telemetry forensics."
        ),
        "Profile C (Completely Irrelevant Domain Match)": (
            "Professional Chef with extensive background in high-volume restaurant management, "
            "menu design, food preparation, and culinary asset procurement. Dedicated to kitchen "
            "hygiene compliance, culinary staff scheduling, and inventory overhead control."
        )
    }

    print("----------------------------------------------------------------------------------")
    print(f"TARGET REQUIREMENTS ARRAY:\n\"{telenor_job_profile}\"")
    print("----------------------------------------------------------------------------------\n")
    print("[COMPUTE] Running Hyperspace Coordinate Computations...\n")

    # Loop over the test metrics and output scalar indexes
    for name, text in test_resumes.items():
        print(f"--> Testing Matrix Node: {name}")
        
        # Execute mathematical vector mapping processing
        start_eval = time.time()
        similarity_index = engine.evaluate_alignment(text, telenor_job_profile)
        execution_delta = (time.time() - start_eval) * 1000  # Convert to milliseconds
        
        # Convert metric index to standard percentage format
        percentage_match = similarity_index * 100
        
        print(f"  |-- Compute Latency   : {execution_delta:.2f} ms")
        print(f"  |-- Math Cosine Score : {similarity_index:.4f}")
        print(f"  |-- Contextual Match  : {percentage_match:.2f}%")
        
        # Output evaluation states based on mathematical thresholds
        if similarity_index >= 0.60:
            print("  |-- Allocation State  : [PASS] HIGH SEMANTIC ALIGNMENT (PASS TO INTERVIEW)")
        elif 0.30 <= similarity_index < 0.60:
            print("  |-- Allocation State  : [REVIEW] PARTIAL PROXIMITY MIX (MANUAL PORTFOLIO AUDIT)")
        else:
            print("  |-- Allocation State  : [REJECT] NON-COMPLIANT DATA LAYER (AUTO-REJECT)")
        print()

def load_engine():
    # Helper wrapper layer to decouple loading sequence loops
    return SemanticMappingEngine()

if __name__ == "__main__":
    run_phase_2_benchmark()
