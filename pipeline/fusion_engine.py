"""
SecurTalent – Module 4: Multi-Layer Mathematical Fusion Engine
===============================================================

Synthesises the metrics from the defensive, semantic, and identity
modules into a unified evaluation matrix by calculating an empirical
Authenticity Coefficient (C_a).

Mathematical Model
------------------
The base coefficient aggregates the semantic alignment score and the
provenance confidence score using configurable operational weights:

    C_a = w_1 * S_similarity + w_2 * G_provenance

Subject to the normalisation constraint:

    sum(w_i) = 1.0

If Layer 1 (Integrity Guard) triggers a security violation (CR < 1.1),
a severe dynamic adversarial penalty factor is applied multiplicatively:

    C_final = C_a * P_adv

Where P_adv = 0.20 by default (80% score reduction).

The engine also computes a qualitative risk classification tier based
on the final coefficient value:

    C_final >= 0.75  -->  LOW RISK       (Strong candidate alignment)
    C_final >= 0.50  -->  MODERATE RISK  (Partial alignment, review needed)
    C_final >= 0.30  -->  HIGH RISK      (Weak alignment or integrity flags)
    C_final <  0.30  -->  CRITICAL RISK  (Likely adversarial or fabricated)

Dependencies
------------
- Python standard library only (no external dependencies)
"""

from typing import Any


class FusionEngine:
    """
    Multi-Layer Mathematical Fusion Engine.

    Aggregates analytical metrics from all upstream detection modules
    to compute the final Authenticity Coefficient (C_final) and a
    qualitative risk classification tier.
    """

    # ── Risk Classification Thresholds ───────────────────────────────
    TIER_LOW = 0.75
    TIER_MODERATE = 0.50
    TIER_HIGH = 0.30

    # ── Risk Tier Labels ─────────────────────────────────────────────
    RISK_LABELS = {
        "LOW": "LOW RISK - Strong candidate alignment detected.",
        "MODERATE": "MODERATE RISK - Partial alignment. Manual review recommended.",
        "HIGH": "HIGH RISK - Weak alignment or integrity anomalies flagged.",
        "CRITICAL": "CRITICAL RISK - Likely adversarial manipulation or fabricated claims.",
    }

    def __init__(
        self,
        weight_similarity: float = 0.60,
        weight_provenance: float = 0.40,
        adversarial_penalty: float = 0.20,
    ):
        """
        Initialise the Fusion Engine with operational weight parameters.

        Parameters
        ----------
        weight_similarity : float
            Weight assigned to semantic alignment score (w_1).
        weight_provenance : float
            Weight assigned to provenance confidence score (w_2).
        adversarial_penalty : float
            Multiplicative penalty factor applied when Layer 1 detects
            a sub-visual whitelisting attack. Default is 0.20 (80% penalty).
        """
        # Normalise weights to enforce sum(w_i) = 1.0
        total = weight_similarity + weight_provenance
        if total > 0:
            self.w1 = weight_similarity / total
            self.w2 = weight_provenance / total
        else:
            self.w1 = 0.5
            self.w2 = 0.5

        self.p_adv = adversarial_penalty

        # Internal cache for the last fusion result
        self._last_result: dict[str, Any] = {}

    def calculate_final_score(
        self,
        similarity: float,
        provenance: float,
        is_compromised: bool,
    ) -> float:
        """
        Synthesise metrics from all upstream modules into C_final.

        Parameters
        ----------
        similarity : float
            Semantic alignment score S_similarity from Layer 2 (NLP Engine).
            Expected range: [0.0, 1.0].
        provenance : float
            Provenance confidence score G_provenance from Layer 3.
            Expected range: [0.0, 1.0].
        is_compromised : bool
            True if Layer 1 (Integrity Guard) detected a sub-visual
            whitelisting attack (CR < threshold).

        Returns
        -------
        float
            Final Authenticity Coefficient C_final in [0.0, 1.0].
        """
        # ── Clamp inputs to valid range ──────────────────────────────
        similarity = max(0.0, min(1.0, similarity))
        provenance = max(0.0, min(1.0, provenance))

        # ── Compute base Authenticity Coefficient (C_a) ──────────────
        c_a = (self.w1 * similarity) + (self.w2 * provenance)

        # ── Apply adversarial penalty if Layer 1 triggered ───────────
        if is_compromised:
            c_final = c_a * self.p_adv
        else:
            c_final = c_a

        c_final = round(c_final, 4)

        # ── Classify risk tier ───────────────────────────────────────
        risk_tier = self._classify_risk(c_final)

        # ── Cache detailed result for dashboard introspection ────────
        self._last_result = {
            "s_similarity": round(similarity, 4),
            "g_provenance": round(provenance, 4),
            "w1": round(self.w1, 4),
            "w2": round(self.w2, 4),
            "c_a_base": round(c_a, 4),
            "is_compromised": is_compromised,
            "penalty_applied": self.p_adv if is_compromised else 1.0,
            "c_final": c_final,
            "risk_tier": risk_tier,
            "risk_label": self.RISK_LABELS[risk_tier],
        }

        return c_final

    def _classify_risk(self, c_final: float) -> str:
        """
        Map the final coefficient to a qualitative risk tier.

        Parameters
        ----------
        c_final : float
            Final Authenticity Coefficient.

        Returns
        -------
        str
            One of: "LOW", "MODERATE", "HIGH", "CRITICAL".
        """
        if c_final >= self.TIER_LOW:
            return "LOW"
        elif c_final >= self.TIER_MODERATE:
            return "MODERATE"
        elif c_final >= self.TIER_HIGH:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_last_result(self) -> dict[str, Any]:
        """Return the detailed fusion result from the last calculation."""
        return self._last_result.copy()

    def __repr__(self) -> str:
        return (
            f"FusionEngine("
            f"w1={self.w1:.2f}, "
            f"w2={self.w2:.2f}, "
            f"P_adv={self.p_adv})"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Self-test / usage example
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("SecurTalent - Module 4: Fusion Engine - Self-Test")
    print("=" * 70)

    engine = FusionEngine()
    print(f"\nEngine Configuration: {engine}")
    print(f"  w1 (Semantic)   = {engine.w1:.2f}")
    print(f"  w2 (Provenance) = {engine.w2:.2f}")
    print(f"  P_adv           = {engine.p_adv}")

    # ── Test 1: Strong candidate, no compromise ──────────────────────
    print("\n[1] Strong candidate (S=0.85, G=0.90, clean)...")
    score = engine.calculate_final_score(0.85, 0.90, is_compromised=False)
    result = engine.get_last_result()
    print(f"    C_a (base)   = {result['c_a_base']}")
    print(f"    C_final      = {score}")
    print(f"    Risk Tier    = {result['risk_tier']}")
    print(f"    Risk Label   = {result['risk_label']}")
    assert result['risk_tier'] == "LOW", f"Expected LOW, got {result['risk_tier']}"
    print("    PASSED")

    # ── Test 2: Moderate candidate, no compromise ────────────────────
    print("\n[2] Moderate candidate (S=0.55, G=0.60, clean)...")
    score = engine.calculate_final_score(0.55, 0.60, is_compromised=False)
    result = engine.get_last_result()
    print(f"    C_a (base)   = {result['c_a_base']}")
    print(f"    C_final      = {score}")
    print(f"    Risk Tier    = {result['risk_tier']}")
    assert result['risk_tier'] == "MODERATE", f"Expected MODERATE, got {result['risk_tier']}"
    print("    PASSED")

    # ── Test 3: Strong candidate WITH compromise (SVWA penalty) ──────
    print("\n[3] Strong candidate WITH SVWA attack (S=0.85, G=0.90, COMPROMISED)...")
    score = engine.calculate_final_score(0.85, 0.90, is_compromised=True)
    result = engine.get_last_result()
    print(f"    C_a (base)   = {result['c_a_base']}")
    print(f"    Penalty      = x{result['penalty_applied']}")
    print(f"    C_final      = {score}")
    print(f"    Risk Tier    = {result['risk_tier']}")
    assert result['risk_tier'] == "CRITICAL", f"Expected CRITICAL with penalty, got {result['risk_tier']}"
    assert score < 0.20, f"Expected < 0.20 after penalty, got {score}"
    print("    PASSED (80% penalty correctly applied)")

    # ── Test 4: Zero inputs ──────────────────────────────────────────
    print("\n[4] Zero inputs (S=0.0, G=0.0, clean)...")
    score = engine.calculate_final_score(0.0, 0.0, is_compromised=False)
    result = engine.get_last_result()
    print(f"    C_final      = {score}")
    print(f"    Risk Tier    = {result['risk_tier']}")
    assert score == 0.0, f"Expected 0.0, got {score}"
    assert result['risk_tier'] == "CRITICAL"
    print("    PASSED")

    # ── Test 5: Custom weights ───────────────────────────────────────
    print("\n[5] Custom weights (w1=0.8, w2=0.2)...")
    custom_engine = FusionEngine(weight_similarity=0.8, weight_provenance=0.2)
    print(f"    Config: {custom_engine}")
    score = custom_engine.calculate_final_score(0.90, 0.40, is_compromised=False)
    result = custom_engine.get_last_result()
    print(f"    C_a (base)   = {result['c_a_base']}")
    print(f"    C_final      = {score}")
    # With w1=0.8, w2=0.2: C_a = 0.8*0.9 + 0.2*0.4 = 0.72 + 0.08 = 0.80
    assert abs(score - 0.80) < 0.01, f"Expected ~0.80, got {score}"
    print("    PASSED")

    # ── Test 6: Risk tier boundary verification ──────────────────────
    print("\n[6] Risk tier boundary checks...")
    boundaries = [
        (0.76, "LOW"),
        (0.75, "LOW"),
        (0.74, "MODERATE"),
        (0.50, "MODERATE"),
        (0.49, "HIGH"),
        (0.30, "HIGH"),
        (0.29, "CRITICAL"),
        (0.00, "CRITICAL"),
    ]
    for c_val, expected_tier in boundaries:
        tier = engine._classify_risk(c_val)
        status = "PASS" if tier == expected_tier else "FAIL"
        print(f"    C={c_val:.2f} -> {tier:>10s}  [{status}]")
        assert tier == expected_tier, f"Boundary check failed for C={c_val}"
    print("    ALL PASSED")

    print()
    print("=" * 70)
    print("Module 4 self-test complete.")
    print("=" * 70)
