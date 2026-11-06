"""Figure 1 - XQFL framework diagram.

Figure 1 is a hand-designed schematic (not plotted from data). The master file
is paper_assets/figure1_architecture_source.png. This script crops the baked-in
"Figure 1. ..." title band off the top — pandoc adds its own numbered caption —
and writes the manuscript-ready image to results/outputs/figure1_architecture.png.

Faithful to XQFL_paper.md Sections 3-6: DATA (shared PCA -> 4-client partition),
LOCAL (per-client encode / VQC / local Q-SHAP), SERVER (FedAvg -> global VQC +
Track A/B gate -> global Q-SHAP; a parallel non-federated centralized model gives
the RQ1 reference), ANALYSIS (RQ1-RQ4, each the pair of attribution vectors it
compares). All vectors live in the same PCA(4) space with the same Q-SHAP
estimator (Eqs. 2-6): comparability before comparison.
"""

import os

from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper_assets", "figure1_architecture_source.png")
OUT = os.path.join(ROOT, "results", "outputs", "figure1_architecture.png")

TITLE_BAND_PX = 92          # rows to drop from the top (the "Figure 1. ..." title)
MARGIN = (14, 10, 14, 12)   # (left, top, right, bottom) white padding to add back


def main():
    if not os.path.exists(SRC):
        raise SystemExit(
            f"master image not found: {SRC}\n"
            "Place the hand-designed architecture PNG there (see module docstring)."
        )
    im = Image.open(SRC).convert("RGB")
    w, h = im.size
    im = im.crop((0, TITLE_BAND_PX, w, h))
    im = ImageOps.expand(im, border=MARGIN, fill="white")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    im.save(OUT, dpi=(300, 300))
    print(f"wrote {OUT}  ({im.size[0]}x{im.size[1]})")


if __name__ == "__main__":
    main()
