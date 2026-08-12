from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


APP_TITLE = "HK Smart Plastic Sorter"
APP_ICON = "♻️"
PROJECT_ROOT = Path(__file__).resolve().parent
BEST_PT_PATH = PROJECT_ROOT / "best.pt"


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    display_name: str
    bbox: tuple[int, int, int, int]


RIC_GUIDE = [
    ("1", "PET", "Polyethylene Terephthalate", "Vitasoy water bottles, clear beverage bottles"),
    ("2", "HDPE", "High-Density Polyethylene", "Watson's distilled water large jugs, detergent bottles"),
    ("3", "PVC", "Polyvinyl Chloride", "Some blister packs, pipes, legacy packaging"),
    ("4", "LDPE", "Low-Density Polyethylene", "Squeezable bottles, plastic bags, film wraps"),
    ("5", "PP", "Polypropylene", "Tao Ti tea bottle caps, microwaveable food containers"),
    ("6", "PS", "Polystyrene", "Disposable cups, foam trays, cutlery"),
    ("7", "OTHER", "Other plastics", "Multi-layer bottles, mixed-material containers"),
]

RIC_TO_BANNER = {
    "PET_1": "♻️ PET (Type 1) Detected!",
    "HDPE_2": "♻️ HDPE (Type 2) Detected!",
    "PVC_3": "♻️ PVC (Type 3) Detected!",
    "LDPE_4": "♻️ LDPE (Type 4) Detected!",
    "PP_5": "♻️ PP (Type 5) Detected!",
    "PS_6": "♻️ PS (Type 6) Detected!",
    "OTHER_7": "♻️ OTHER (Type 7) Detected!",
}

RIC_TO_EXAMPLES = {
    "PET_1": "Common for clear drink bottles such as Vitasoy water bottles.",
    "HDPE_2": "Seen in larger water jugs and stronger household containers.",
    "PVC_3": "Less common for drinks; often appears in rigid packaging or non-food plastics.",
    "LDPE_4": "Typical of flexible, squeezable packaging and wraps.",
    "PP_5": "Often used for caps, lids, and microwave-safe containers.",
    "PS_6": "Found in disposable foodware and foam packaging.",
    "OTHER_7": "Used when the plastic is a blend or does not fit the main categories.",
}


def set_page_style() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            :root {
                --bg: #f4f7f3;
                --card: rgba(255, 255, 255, 0.88);
                --text: #10311f;
                --muted: #486154;
                --accent: #0f8b4c;
                --accent-strong: #0a6b3b;
                --line: rgba(16, 49, 31, 0.12);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 139, 76, 0.12), transparent 30%),
                    radial-gradient(circle at top right, rgba(28, 94, 63, 0.08), transparent 24%),
                    linear-gradient(180deg, #f7fbf7 0%, #eff6ef 100%);
                color: var(--text);
            }

            .block-container {
                padding-top: 1.1rem;
                padding-bottom: 2rem;
                max-width: 820px;
            }

            .hero {
                background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(236,247,238,0.92));
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 1.25rem 1.2rem;
                box-shadow: 0 14px 40px rgba(16, 49, 31, 0.08);
                margin-bottom: 1rem;
            }

            .hero h1 {
                font-size: 2rem;
                line-height: 1.05;
                margin-bottom: 0.35rem;
            }

            .hero p {
                color: var(--muted);
                margin-bottom: 0;
                font-size: 0.98rem;
            }

            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 0.9rem;
            }

            .pill {
                background: rgba(15, 139, 76, 0.1);
                border: 1px solid rgba(15, 139, 76, 0.18);
                color: var(--accent-strong);
                border-radius: 999px;
                padding: 0.42rem 0.72rem;
                font-size: 0.82rem;
                font-weight: 600;
            }

            .scan-card {
                background: var(--card);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 1rem;
                box-shadow: 0 10px 28px rgba(16, 49, 31, 0.06);
                margin-top: 0.75rem;
            }

            .section-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.2rem;
            }

            .section-subtitle {
                color: var(--muted);
                margin-bottom: 0.85rem;
                font-size: 0.92rem;
            }

            div[data-testid="stTabs"] {
                margin-top: 0.3rem;
            }

            div[data-testid="stTabList"] {
                gap: 0.25rem;
                background: rgba(255,255,255,0.45);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.3rem;
            }

            button[data-baseweb="tab"] {
                border-radius: 14px !important;
                font-weight: 700 !important;
            }

            .table-wrap {
                overflow-x: auto;
                border-radius: 18px;
                border: 1px solid var(--line);
                background: rgba(255,255,255,0.72);
                box-shadow: 0 10px 24px rgba(16, 49, 31, 0.05);
            }

            table {
                width: 100%;
                border-collapse: collapse;
                min-width: 640px;
            }

            th, td {
                padding: 0.8rem 0.85rem;
                text-align: left;
                vertical-align: top;
                border-bottom: 1px solid rgba(16, 49, 31, 0.08);
                font-size: 0.92rem;
            }

            th {
                background: rgba(15, 139, 76, 0.08);
                color: var(--accent-strong);
                font-size: 0.84rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            tr:last-child td {
                border-bottom: none;
            }

            .about-box {
                background: var(--card);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 1rem 1rem 0.95rem;
                box-shadow: 0 10px 28px rgba(16, 49, 31, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model() -> YOLO | None:
    if YOLO is None:
        return None
    if BEST_PT_PATH.exists():
        return YOLO(str(BEST_PT_PATH))
    return None


def preprocess_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert the input image into a contrast-enhanced edge map for inspection."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = np.array(image)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[-1] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 40, 120)
    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edge_rgb


def predict_yolo(image: np.ndarray) -> tuple[np.ndarray, DetectionResult]:
    """Placeholder YOLOv8 inference.

    Replace this with real model inference when your `best.pt` is ready.
    Example insertion point:
        model = YOLO("best.pt")
        results = model.predict(source=image, imgsz=640, conf=0.25)
    """
    height, width = image.shape[:2]
    x1 = max(10, width // 7)
    y1 = max(12, height // 3)
    x2 = min(width - 10, x1 + max(120, width // 2))
    y2 = min(height - 12, y1 + max(100, height // 4))

    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (27, 150, 90), 3)
    cv2.rectangle(annotated, (x1, max(0, y1 - 32)), (min(width - 1, x1 + 220), y1), (27, 150, 90), -1)
    cv2.putText(
        annotated,
        "PP_5 0.94",
        (x1 + 10, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    result = DetectionResult(
        label="PP_5",
        confidence=0.94,
        display_name="PP (Type 5)",
        bbox=(x1, y1, x2, y2),
    )
    return annotated, result


def render_scan_tab(model: YOLO | None) -> None:
    st.markdown('<div class="scan-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scan a bottle bottom</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Use the camera or upload a gallery image. The app highlights embossed resin codes before mock detection.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("💡 **Tip:** Point the camera at the *bottom* of the bottle. Ensure there is some light casting a shadow on the embossed code.")

    camera_image = st.camera_input("Take a photo of the bottle bottom")
    gallery_image = st.file_uploader("Or upload from gallery", type=["png", "jpg", "jpeg", "webp"])

    uploaded_image: Image.Image | None = None
    if camera_image is not None:
        uploaded_image = Image.open(camera_image)
    elif gallery_image is not None:
        uploaded_image = Image.open(gallery_image)

    if uploaded_image is not None:
        raw_rgb = np.array(uploaded_image.convert("RGB"))
        enhanced_rgb = preprocess_image(uploaded_image)
        annotated_rgb, detection = predict_yolo(enhanced_rgb)

        banner = RIC_TO_BANNER.get(detection.label, f"♻️ {detection.display_name} Detected!")
        st.success(banner)
        st.caption(f"Mock confidence: {detection.confidence:.2f} | Bounding box: {detection.bbox}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Raw Image**")
            st.image(raw_rgb, use_container_width=True)
        with right:
            st.markdown("**Enhanced Edge Map**")
            st.image(enhanced_rgb, use_container_width=True)

        st.markdown("**YOLO Detection Preview**")
        st.image(annotated_rgb, use_container_width=True)

        st.info(
            RIC_TO_EXAMPLES.get(
                detection.label,
                "This mock output is ready to be replaced with your real YOLOv8 model weights.",
            )
        )
    else:
        st.info("Take a photo or upload an image to begin. The app will keep this space empty until an image is available.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_guide_tab() -> None:
    st.markdown('<div class="about-box">', unsafe_allow_html=True)
    st.markdown("## Plastic RIC Guide")
    st.markdown("A quick reference for the 7 Resin Identification Codes commonly seen on plastic packaging in Hong Kong.")

    table_html = [
        "<div class='table-wrap'><table>",
        "<thead><tr><th>Code</th><th>Type</th><th>Full Name</th><th>Hong Kong example</th></tr></thead><tbody>",
    ]
    for code, short_name, full_name, example in RIC_GUIDE:
        table_html.append(
            f"<tr><td><strong>{code}</strong></td><td>{short_name}</td><td>{full_name}</td><td>{example}</td></tr>"
        )
    table_html.append("</tbody></table></div>")
    st.markdown("".join(table_html), unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-top: 0.9rem; color: #486154; font-size: 0.94rem;">
        <strong>Practical reading tip:</strong> on transparent or translucent bottles, the embossed code is often easiest to see when light hits it from the side and creates a shadow edge.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_about_tab() -> None:
    st.markdown('<div class="about-box">', unsafe_allow_html=True)
    st.markdown("## Optimizing Computer Vision for Transparent and Translucent Plastic Bottle Classification in Hong Kong")
    st.markdown(
        "This project helps citizens identify transparent plastic bottles by reading the embossed Resin Identification Codes (RIC) on bottle bottoms using computer vision."
    )
    st.markdown("**Developer:** Chan Pui Yan Nevalle, Coventry University BSc (Hons) Computing")
    st.markdown("### Methodology")
    st.markdown(
        """
        - **CLAHE** improves local contrast so faint embossed markings stand out more clearly.
        - **Canny Edge Detection** converts subtle grooves and raised characters into stronger outlines.
        - The cleaned edge map makes it easier for **YOLOv8** to localize the code region that might otherwise look nearly invisible on transparent plastic.
        """
    )
    st.caption("Insert your actual `best.pt` later at the placeholder in `predict_yolo()` or by wiring in a real inference path.")
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    set_page_style()
    model = load_model()

    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_ICON} {APP_TITLE}</h1>
            <p>Scan transparent bottles, enhance the embossed code, and preview a YOLO-style detection flow built for Hong Kong recycling use cases.</p>
            <div class="pill-row">
                <span class="pill">Mobile-friendly centered layout</span>
                <span class="pill">OpenCV preprocessing</span>
                <span class="pill">YOLOv8-ready placeholder</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scan_tab, guide_tab, about_tab = st.tabs(["📷 Scan", "📖 Guide", "ℹ️ About"])

    with scan_tab:
        render_scan_tab(model)

    with guide_tab:
        render_guide_tab()

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()