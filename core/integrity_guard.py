"""
SecurTalent – Module 1: Integrity Guard
========================================

Sub-visual whitelisting attack detection via low-level glyph contrast auditing.

This module implements deterministic detection of adversarial text hidden within
PDF documents by rendering at near-zero contrast against the page background.
Attackers exploit this technique ("sub-visual whitelisting") to inject invisible
keywords—skills, certifications, employer names—that are parsed by automated
systems but invisible to human reviewers.

Detection is performed by computing the W3C relative luminance and WCAG 2.1
contrast ratio for every individual character glyph extracted from the PDF's
content stream, then flagging any glyph whose contrast ratio against the assumed
page background falls below a configurable threshold.

Mathematical Foundations
------------------------
1. **sRGB Linearisation** (IEC 61966-2-1):
   For each sRGB channel value C_srgb ∈ [0, 1]:

       C_linear = C_srgb / 12.92                       if C_srgb <= 0.04045
       C_linear = ((C_srgb + 0.055) / 1.055) ^ 2.4     otherwise

2. **W3C Relative Luminance** (WCAG 2.1 §1.4.3):
       L = 0.2126 * R_linear + 0.7152 * G_linear + 0.0722 * B_linear

   where L ∈ [0, 1], with 0 = darkest black and 1 = pure white.

3. **Contrast Ratio** (WCAG 2.1):
       CR = (L_lighter + 0.05) / (L_darker + 0.05)

   where L_lighter = max(L_text, L_bg) and L_darker = min(L_text, L_bg).
   A ratio of 1.0 means identical luminance (invisible). Standard WCAG AA
   requires CR >= 4.5 for normal text; this module flags text with CR below
   a much lower threshold (default 1.1), indicating near-invisible rendering.

Dependencies
------------
- pdfplumber (PDF content stream extraction)
- Python standard library only for computation (no numpy needed here)
"""

import pdfplumber
from typing import Any


def calculate_relative_luminance(color_tuple: tuple[int, int, int]) -> float:
    """
    Compute the W3C relative luminance of an sRGB colour.

    Implements the two-step process defined in WCAG 2.1 §1.4.3:
      1. Linearise each sRGB channel via the IEC 61966-2-1 transfer function.
      2. Apply the ITU-R BT.709 luminance coefficients.

    Parameters
    ----------
    color_tuple : tuple[int, int, int]
        An (R, G, B) triplet with each component in [0, 255].

    Returns
    -------
    float
        Relative luminance L ∈ [0.0, 1.0].
        0.0 corresponds to absolute black; 1.0 to reference white.

    Examples
    --------
    >>> calculate_relative_luminance((0, 0, 0))
    0.0
    >>> calculate_relative_luminance((255, 255, 255))
    1.0
    >>> round(calculate_relative_luminance((254, 254, 254)), 4)  # near-white
    0.9911
    """
    # --- Step 1: sRGB → linear RGB (IEC 61966-2-1 inverse EOTF) ---
    linearised = []
    for channel_8bit in color_tuple:
        # Normalise from [0, 255] to [0, 1]
        c_srgb = channel_8bit / 255.0

        # Apply piecewise linearisation
        if c_srgb <= 0.04045:
            c_linear = c_srgb / 12.92
        else:
            c_linear = ((c_srgb + 0.055) / 1.055) ** 2.4

        linearised.append(c_linear)

    # --- Step 2: ITU-R BT.709 weighted sum ---
    # L = 0.2126·R + 0.7152·G + 0.0722·B
    r_lin, g_lin, b_lin = linearised
    luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    return luminance


def _compute_contrast_ratio(l_text: float, l_background: float) -> float:
    """
    Compute the WCAG 2.1 contrast ratio between two luminance values.

    Parameters
    ----------
    l_text : float
        Relative luminance of the foreground (text).
    l_background : float
        Relative luminance of the background.

    Returns
    -------
    float
        Contrast ratio CR ∈ [1.0, 21.0].
    """
    l_lighter = max(l_text, l_background)
    l_darker = min(l_text, l_background)
    return (l_lighter + 0.05) / (l_darker + 0.05)


def _extract_glyph_color(char: dict[str, Any]) -> tuple[int, int, int]:
    """
    Extract an (R, G, B) colour tuple from a pdfplumber character object.

    pdfplumber stores stroking/non-stroking colours in various formats
    depending on the PDF colour space (DeviceRGB, DeviceGray, etc.).
    This function normalises to 8-bit sRGB.

    Parameters
    ----------
    char : dict
        A pdfplumber character dictionary (from page.chars).

    Returns
    -------
    tuple[int, int, int]
        Normalised (R, G, B) in [0, 255].
    """
    # pdfplumber exposes non-stroking colour (fill) as 'non_stroking_color'
    color = char.get("non_stroking_color")

    if color is None:
        # No explicit colour → default to black (fully visible)
        return (0, 0, 0)

    # Handle different colour-space representations
    if isinstance(color, (list, tuple)):
        if len(color) == 3:
            # DeviceRGB – values are floats in [0, 1]
            return (
                int(round(color[0] * 255)),
                int(round(color[1] * 255)),
                int(round(color[2] * 255)),
            )
        elif len(color) == 1:
            # DeviceGray – single float in [0, 1] (0 = black, 1 = white)
            gray_8bit = int(round(color[0] * 255))
            return (gray_8bit, gray_8bit, gray_8bit)
        elif len(color) == 4:
            # DeviceCMYK – approximate conversion to sRGB
            c, m, y, k = color
            r = int(round(255 * (1 - c) * (1 - k)))
            g = int(round(255 * (1 - m) * (1 - k)))
            b = int(round(255 * (1 - y) * (1 - k)))
            return (r, g, b)

    # Fallback for unexpected formats → assume black
    return (0, 0, 0)


def audit_document_integrity(
    pdf_path: str,
    contrast_threshold: float = 1.1,
) -> dict[str, Any]:
    """
    Audit a PDF document for sub-visual whitelisting attacks.

    Opens the PDF with pdfplumber and inspects every character glyph on every
    page. For each glyph, the text colour is extracted and its contrast ratio
    against an assumed white background (L = 1.0) is computed. Glyphs with
    a contrast ratio below ``contrast_threshold`` are classified as adversarial
    (sub-visual).

    Parameters
    ----------
    pdf_path : str
        Filesystem path to the PDF document under analysis.
    contrast_threshold : float, optional
        Maximum contrast ratio to consider adversarial. Characters with
        CR < contrast_threshold are flagged. Default is 1.1, corresponding
        to text that is nearly indistinguishable from a white background.

    Returns
    -------
    dict[str, Any]
        Audit result dictionary with the following keys:

        - ``is_compromised`` (bool):
            True if at least one adversarial glyph was detected.
        - ``anomaly_ratio`` (float):
            Fraction of total characters flagged as adversarial [0.0, 1.0].
        - ``adversarial_payloads`` (list[dict]):
            List of flagged glyphs, each containing:
              - ``char``: the character string
              - ``page``: 1-indexed page number
              - ``color_rgb``: the (R, G, B) tuple
              - ``contrast_ratio``: computed CR value
              - ``position``: dict with ``x0``, ``y0``, ``x1``, ``y1``
        - ``cleaned_text`` (str):
            Full document text with adversarial characters removed.

    Raises
    ------
    FileNotFoundError
        If ``pdf_path`` does not exist.
    pdfplumber.exceptions.PDFSyntaxError
        If the file is not a valid PDF.

    Notes
    -----
    The assumed background luminance of 1.0 (pure white) is a conservative
    default. For production use on non-white-background PDFs, background
    colour should be sampled per-page or per-region.
    """
    # Background luminance assumption: pure white (L = 1.0)
    L_BACKGROUND = 1.0

    adversarial_payloads: list[dict[str, Any]] = []
    clean_chars: list[str] = []
    total_char_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_number = page_idx + 1  # 1-indexed for human readability

            chars = page.chars
            if not chars:
                continue

            for char in chars:
                total_char_count += 1
                text = char.get("text", "")

                # --- Extract glyph colour and compute luminance ---
                color_rgb = _extract_glyph_color(char)
                l_text = calculate_relative_luminance(color_rgb)

                # --- Compute contrast ratio against white background ---
                contrast_ratio = _compute_contrast_ratio(l_text, L_BACKGROUND)

                if contrast_ratio < contrast_threshold:
                    # Sub-visual glyph detected → flag as adversarial
                    adversarial_payloads.append(
                        {
                            "char": text,
                            "page": page_number,
                            "color_rgb": color_rgb,
                            "contrast_ratio": round(contrast_ratio, 6),
                            "position": {
                                "x0": char.get("x0"),
                                "y0": char.get("top"),
                                "x1": char.get("x1"),
                                "y1": char.get("bottom"),
                            },
                        }
                    )
                else:
                    # Legitimate glyph → include in cleaned text
                    clean_chars.append(text)

    # --- Compile audit results ---
    anomaly_ratio = (
        len(adversarial_payloads) / total_char_count
        if total_char_count > 0
        else 0.0
    )

    return {
        "is_compromised": len(adversarial_payloads) > 0,
        "anomaly_ratio": round(anomaly_ratio, 6),
        "adversarial_payloads": adversarial_payloads,
        "cleaned_text": "".join(clean_chars),
    }


# ---------------------------------------------------------------------------
# Self-test / usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    print("=" * 70)
    print("SecurTalent – Module 1: Integrity Guard – Self-Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Unit test: verify luminance calculations against known values
    # ------------------------------------------------------------------
    print("\n[1] Luminance unit tests...")

    assert calculate_relative_luminance((0, 0, 0)) == 0.0, "Black must be 0.0"
    assert calculate_relative_luminance((255, 255, 255)) == 1.0, "White must be 1.0"

    # Near-white (#fefefe) should have very high luminance
    l_near_white = calculate_relative_luminance((254, 254, 254))
    assert l_near_white > 0.98, f"Near-white luminance should be >0.98, got {l_near_white}"

    print("   ✓ Black  → L = 0.0")
    print("   ✓ White  → L = 1.0")
    print(f"   ✓ #fefefe → L = {l_near_white:.6f}")

    # ------------------------------------------------------------------
    # Unit test: contrast ratio
    # ------------------------------------------------------------------
    print("\n[2] Contrast ratio unit tests...")

    cr_bw = _compute_contrast_ratio(0.0, 1.0)
    assert cr_bw == 21.0, f"Black-on-white CR should be 21.0, got {cr_bw}"
    print(f"   ✓ Black on white → CR = {cr_bw}")

    cr_ww = _compute_contrast_ratio(1.0, 1.0)
    assert cr_ww == 1.0, f"White-on-white CR should be 1.0, got {cr_ww}"
    print(f"   ✓ White on white → CR = {cr_ww}")

    cr_near = _compute_contrast_ratio(l_near_white, 1.0)
    print(f"   ✓ #fefefe on white → CR = {cr_near:.6f} (should be ~1.0)")

    # ------------------------------------------------------------------
    # PDF audit (if a test file is provided)
    # ------------------------------------------------------------------
    print("\n[3] PDF audit test...")

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_pdf = sys.argv[1]
        print(f"   Auditing: {test_pdf}")
        result = audit_document_integrity(test_pdf)
        print(f"   Compromised : {result['is_compromised']}")
        print(f"   Anomaly ratio : {result['anomaly_ratio']:.4%}")
        print(f"   Payloads found: {len(result['adversarial_payloads'])}")
        if result["adversarial_payloads"]:
            print("   First 5 adversarial glyphs:")
            for p in result["adversarial_payloads"][:5]:
                print(f"     '{p['char']}' | page {p['page']} | "
                      f"RGB {p['color_rgb']} | CR {p['contrast_ratio']:.4f}")
        print(f"   Cleaned text length: {len(result['cleaned_text'])} chars")
    else:
        print("   No test PDF provided. Skipping live audit.")
        print("   Usage: python integrity_guard.py <path_to_test.pdf>")
        print()
        print("   ── Creating a test PDF with hidden text ──")
        print("   To generate a test PDF with white-on-white text, run:")
        print()
        print("     pip install reportlab")
        print("     python -c \"")
        print("       from reportlab.lib.pagesizes import letter")
        print("       from reportlab.pdfgen import canvas")
        print("       c = canvas.Canvas('test_adversarial.pdf', pagesize=letter)")
        print("       # Visible black text")
        print("       c.setFillColorRGB(0, 0, 0)")
        print("       c.drawString(72, 700, 'John Doe - Software Engineer')")
        print("       # Hidden white-on-white text (adversarial)")
        print("       c.setFillColorRGB(1, 1, 1)")
        print("       c.drawString(72, 680, 'PhD MIT Stanford 10x Engineer')")
        print("       c.save()")
        print("     \"")
        print()
        print("   Then re-run: python integrity_guard.py test_adversarial.pdf")

    print()
    print("=" * 70)
    print("Module 1 self-test complete.")
    print("=" * 70)
