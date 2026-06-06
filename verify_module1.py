try:
    from core.integrity_guard import audit_document_integrity, calculate_relative_luminance
    print("Module 1 ready")
except Exception as e:
    print(f"Failed to import integrity_guard: {e}")
