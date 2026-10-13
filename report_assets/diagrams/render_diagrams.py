#!/usr/bin/env python3
"""
render_diagrams.py
------------------
Render all 7 architecture, workflow, and UML diagrams (Figs 1-7)
from report_assets/diagrams/source/ into publication-ready PNG and vector SVG
under report_assets/diagrams/exports/.
"""

import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = REPO_ROOT / "report_assets" / "diagrams" / "source"
EXPORT_DIR = REPO_ROOT / "report_assets" / "diagrams" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

DIAGRAMS = [
    {
        "id": "fig1",
        "name": "fig1_system_architecture",
        "file": "fig1_system_architecture.mmd",
        "type": "mermaid",
        "title": "Fig. 1 — Overall System Architecture"
    },
    {
        "id": "fig2",
        "name": "fig2_data_preprocessing_workflow",
        "file": "fig2_data_preprocessing_workflow.mmd",
        "type": "mermaid",
        "title": "Fig. 2 — Data Preprocessing Workflow"
    },
    {
        "id": "fig3",
        "name": "fig3_spatial_analysis_workflow",
        "file": "fig3_spatial_analysis_workflow.mmd",
        "type": "mermaid",
        "title": "Fig. 3 — Spatial Analysis Workflow"
    },
    {
        "id": "fig4",
        "name": "fig4_sequential_mining_workflow",
        "file": "fig4_sequential_mining_workflow.mmd",
        "type": "mermaid",
        "title": "Fig. 4 — Sequential Mining Workflow"
    },
    {
        "id": "fig5",
        "name": "fig5_uml_use_case",
        "file": "fig5_uml_use_case.puml",
        "type": "plantuml",
        "title": "Fig. 5 — UML Use-Case Model"
    },
    {
        "id": "fig6",
        "name": "fig6_uml_activity",
        "file": "fig6_uml_activity.puml",
        "type": "plantuml",
        "title": "Fig. 6 — UML Activity Diagram"
    },
    {
        "id": "fig7",
        "name": "fig7_uml_sequence",
        "file": "fig7_uml_sequence.puml",
        "type": "plantuml",
        "title": "Fig. 7 — UML Sequence Diagram"
    },
]


def render_diagram(diag_type: str, output_format: str, source_text: str) -> bytes:
    """Render diagram via Kroki service with retries and timeout."""
    url = f"https://kroki.io/{diag_type}/{output_format}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "text/plain; charset=utf-8"
    }
    data = source_text.encode("utf-8")

    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    return resp.read()
                else:
                    raise RuntimeError(f"HTTP {resp.status}")
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed after 3 attempts: {last_err}")


def main():
    print(f"Rendering {len(DIAGRAMS)} diagrams from {SOURCE_DIR} to {EXPORT_DIR}...")
    success_count = 0
    failures = []

    for d in DIAGRAMS:
        src_path = SOURCE_DIR / d["file"]
        if not src_path.exists():
            print(f"[-] Missing source file: {src_path}")
            failures.append((d["name"], "Missing source file"))
            continue

        source_text = src_path.read_text(encoding="utf-8")
        print(f"[*] Rendering {d['title']} ({d['type']})...")

        # 1. Render PNG
        png_out = EXPORT_DIR / f"{d['name']}.png"
        try:
            png_bytes = render_diagram(d["type"], "png", source_text)
            png_out.write_bytes(png_bytes)
            print(f"    [+] PNG: {png_out.name} ({len(png_bytes):,} bytes)")
        except Exception as e:
            print(f"    [-] PNG Failed for {d['name']}: {e}")
            failures.append((d["name"] + ".png", str(e)))

        # 2. Render SVG
        svg_out = EXPORT_DIR / f"{d['name']}.svg"
        try:
            svg_bytes = render_diagram(d["type"], "svg", source_text)
            svg_out.write_bytes(svg_bytes)
            print(f"    [+] SVG: {svg_out.name} ({len(svg_bytes):,} bytes)")
        except Exception as e:
            print(f"    [-] SVG Failed for {d['name']}: {e}")
            failures.append((d["name"] + ".svg", str(e)))

        if png_out.exists() and svg_out.exists():
            success_count += 1

    print(f"\nRender completed: {success_count}/{len(DIAGRAMS)} diagrams exported in PNG and SVG.")
    if failures:
        print(f"Failures recorded: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
