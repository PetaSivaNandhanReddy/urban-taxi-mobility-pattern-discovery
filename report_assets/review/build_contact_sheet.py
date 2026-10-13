#!/usr/bin/env python3
"""
build_contact_sheet.py
----------------------
Build a high-resolution, visually inspectable contact sheet containing all
14 figures (Figs 1-7 diagrams and Figs 8-14 scientific plots) for visual quality review.
Saves to report_assets/review/figure_contact_sheet.png.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPORTS_DIR = REPO_ROOT / "report_assets" / "diagrams" / "exports"
PLOTS_DIR = REPO_ROOT / "report_assets" / "plots"
REVIEW_DIR = REPO_ROOT / "report_assets" / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

FIGURES = [
    # Column 0: Diagrams (Figs 1-7)
    [
        {"id": "Fig. 1", "title": "Overall System Architecture", "path": EXPORTS_DIR / "fig1_system_architecture.png", "type": "Architecture Diagram"},
        {"id": "Fig. 2", "title": "Data Preprocessing Workflow", "path": EXPORTS_DIR / "fig2_data_preprocessing_workflow.png", "type": "Workflow Diagram"},
        {"id": "Fig. 3", "title": "Spatial Analysis Workflow", "path": EXPORTS_DIR / "fig3_spatial_analysis_workflow.png", "type": "Spatial Pipeline"},
        {"id": "Fig. 4", "title": "Sequential Mining Workflow", "path": EXPORTS_DIR / "fig4_sequential_mining_workflow.png", "type": "Sequential Pipeline"},
        {"id": "Fig. 5", "title": "UML Use-Case Model", "path": EXPORTS_DIR / "fig5_uml_use_case.png", "type": "UML Use-Case"},
        {"id": "Fig. 6", "title": "UML Activity Workflow", "path": EXPORTS_DIR / "fig6_uml_activity.png", "type": "UML Activity"},
        {"id": "Fig. 7", "title": "UML Sequence Diagram", "path": EXPORTS_DIR / "fig7_uml_sequence.png", "type": "UML Sequence"},
    ],
    # Column 1: Scientific Plots (Figs 8-14)
    [
        {"id": "Fig. 8", "title": "DBSCAN Hotspot Parameter Sensitivity", "path": PLOTS_DIR / "fig8_clustering_kdistance.png", "type": "Empirical Plot"},
        {"id": "Fig. 9", "title": "Support Threshold Sensitivity (Volume vs Metrics)", "path": PLOTS_DIR / "fig9_support_sensitivity.png", "type": "Sensitivity Plot"},
        {"id": "Fig. 10", "title": "Spatial Grid Resolution Sensitivity", "path": PLOTS_DIR / "fig10_grid_sensitivity.png", "type": "Resolution Trade-off"},
        {"id": "Fig. 11", "title": "Empirical Reference: Baseline vs Tuned", "path": PLOTS_DIR / "fig11_method_comparison.png", "type": "Empirical Benchmark"},
        {"id": "Fig. 12", "title": "Synthetic Ground-Truth Planted Recovery", "path": PLOTS_DIR / "fig12_synthetic_recovery.png", "type": "Objective Ground Truth"},
        {"id": "Fig. 13", "title": "Fleet Expansion Scalability & Runtime", "path": PLOTS_DIR / "fig13_sample_scaling.png", "type": "Computational Scaling"},
        {"id": "Fig. 14", "title": "Diurnal Spatiotemporal Mobility Profiles", "path": PLOTS_DIR / "fig14_temporal_activity.png", "type": "Mobility Dynamics"},
    ]
]


def get_font(size: int, bold: bool = False):
    """Load default truetype font or fallback."""
    font_names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_sheet():
    cell_w = 1200
    cell_h = 750
    header_h = 55
    margin = 30
    top_banner_h = 140

    cols = 2
    rows = 7

    total_w = margin * 3 + cell_w * cols
    total_h = top_banner_h + margin + (cell_h + header_h + margin) * rows

    sheet = Image.new("RGB", (total_w, total_h), color=(248, 250, 252))
    draw = ImageDraw.Draw(sheet)

    # Top banner background
    draw.rectangle([(0, 0), (total_w, top_banner_h)], fill=(15, 23, 42))

    font_banner_title = get_font(32, bold=True)
    font_banner_sub = get_font(18, bold=False)
    font_header_fig = get_font(20, bold=True)
    font_header_sub = get_font(15, bold=False)

    draw.text((margin, 25), "Urban Taxi Mobility Pattern Discovery — IEEE Report Assets Contact Sheet", fill=(255, 255, 255), font=font_banner_title)
    draw.text((margin, 75), "Visual Quality Review: 14 Publication Figures (Figs 1–7 Vector Diagrams, Figs 8–14 Empirical Plots at 300 DPI)", fill=(203, 213, 225), font=font_banner_sub)
    draw.text((margin, 105), "Repository: PetaSivaNandhanReddy/urban-taxi-mobility-pattern-discovery | Branch: feature/taxi-mobility-analytics", fill=(148, 163, 184), font=font_banner_sub)

    for c in range(cols):
        for r in range(rows):
            fig_info = FIGURES[c][r]
            x0 = margin + c * (cell_w + margin)
            y0 = top_banner_h + margin + r * (cell_h + header_h + margin)

            # Cell Header Card
            draw.rectangle([(x0, y0), (x0 + cell_w, y0 + header_h)], fill=(30, 41, 59))
            draw.text((x0 + 15, y0 + 10), f"{fig_info['id']} — {fig_info['title']}", fill=(255, 255, 255), font=font_header_fig)
            draw.text((x0 + 15, y0 + 33), f"Category: {fig_info['type']} | File: {fig_info['path'].name}", fill=(148, 163, 184), font=font_header_sub)

            # Image Card
            img_card_x0 = x0
            img_card_y0 = y0 + header_h
            img_card_x1 = x0 + cell_w
            img_card_y1 = img_card_y0 + cell_h
            draw.rectangle([(img_card_x0, img_card_y0), (img_card_x1, img_card_y1)], fill=(255, 255, 255), outline=(203, 213, 225), width=2)

            img_path = fig_info["path"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as im:
                        im_rgb = im.convert("RGB")
                        # Fit inside cell with padding
                        pad = 20
                        avail_w = cell_w - pad * 2
                        avail_h = cell_h - pad * 2

                        ratio = min(avail_w / im_rgb.width, avail_h / im_rgb.height)
                        new_w = max(1, int(im_rgb.width * ratio))
                        new_h = max(1, int(im_rgb.height * ratio))

                        resized = im_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        paste_x = img_card_x0 + pad + (avail_w - new_w) // 2
                        paste_y = img_card_y0 + pad + (avail_h - new_h) // 2
                        sheet.paste(resized, (paste_x, paste_y))
                except Exception as e:
                    draw.text((img_card_x0 + 40, img_card_y0 + 40), f"Error loading image: {e}", fill=(220, 38, 38), font=font_header_fig)
            else:
                draw.text((img_card_x0 + 40, img_card_y0 + 40), f"IMAGE FILE NOT FOUND: {img_path.name}", fill=(220, 38, 38), font=font_header_fig)

    out_path = REVIEW_DIR / "figure_contact_sheet.png"
    print(f"Saving contact sheet to {out_path} ({total_w}x{total_h}px)...")
    sheet.save(out_path, format="PNG", optimize=True)
    print(f"Contact sheet saved: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_sheet()
