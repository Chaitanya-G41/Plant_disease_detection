"""
GuavaScan — Leaf Disease Detection Dashboard
Streamlit app for Phase 2 presentation.

Architecture: DeiT-tiny Stage 2 model (6 guava classes)
Features:
  - Image upload + preprocessing
  - Top-3 predictions with confidence bars
  - Attention map heatmap overlay
  - Hardcoded RAG-style treatment advice (RAG integration pending)

Run locally:
    streamlit run app/app.py

Run on Colab (after cloning repo and downloading model weights):
    !streamlit run app/app.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether run from repo root or from app/ subdirectory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import get_attention_maps
from src.preprocess import get_transforms, get_inverse_transform

# ── Model weight paths ─────────────────────────────────────────────────────────
STAGE1_PTH = os.path.join(ROOT, "models", "stage1", "stage1_best.pth")
STAGE2_PTH = os.path.join(ROOT, "models", "stage2", "stage2_best.pth")

# ── Class definitions ──────────────────────────────────────────────────────────
# Order MUST match the training class_to_idx from ImageFolder (alphabetical)
GUAVA_CLASSES = [
    "Guava_anthracnose",
    "Guava_healthy",
    "Guava_insect_bite",
    "Guava_multiple",
    "Guava_scorch",
    "Guava_yld",
]

DISPLAY_NAMES = {
    "Guava_anthracnose": "Anthracnose",
    "Guava_healthy":     "Healthy",
    "Guava_insect_bite": "Insect Bite",
    "Guava_multiple":    "Multiple Diseases",
    "Guava_scorch":      "Leaf Scorch",
    "Guava_yld":         "Yellow Leaf Disease",
}

SEVERITY = {
    "Guava_anthracnose": ("High",   "#e74c3c"),
    "Guava_healthy":     ("None",   "#27ae60"),
    "Guava_insect_bite": ("Medium", "#f39c12"),
    "Guava_multiple":    ("High",   "#e74c3c"),
    "Guava_scorch":      ("Medium", "#f39c12"),
    "Guava_yld":         ("High",   "#e74c3c"),
}

# ── Hardcoded treatment advice (RAG placeholder) ───────────────────────────────
DISEASE_ADVICE = {
    "Guava_anthracnose": {
        "overview": (
            "Anthracnose is a fungal disease caused by Colletotrichum gloeosporioides. "
            "It produces dark, sunken lesions on leaves, stems, and fruits, and thrives "
            "in warm, humid conditions."
        ),
        "symptoms": [
            "Dark brown to black irregular spots on leaf surface",
            "Lesions with yellow halo in early stages",
            "Spots coalesce and cause premature leaf drop",
            "Sunken, dark lesions on fruits in severe cases",
        ],
        "treatment": [
            "Apply copper-based fungicide (Copper oxychloride 0.3%) every 10–14 days",
            "Use Mancozeb 0.2% or Carbendazim 0.1% as alternatives",
            "Remove and destroy all infected plant material immediately",
            "Avoid overhead irrigation — switch to drip irrigation",
        ],
        "prevention": [
            "Ensure good air circulation with proper canopy pruning",
            "Avoid wetting foliage during irrigation",
            "Apply preventive fungicide spray during monsoon season",
            "Use disease-free planting material",
        ],
        "note": "⚠️ RAG integration pending — advice based on curated agricultural knowledge base.",
    },
    "Guava_healthy": {
        "overview": (
            "The leaf appears healthy with no visible signs of disease or pest damage. "
            "Continue regular monitoring and preventive care."
        ),
        "symptoms": [
            "No lesions, spots, or discoloration observed",
            "Uniform green color throughout the leaf",
            "Normal leaf texture and shape",
        ],
        "treatment": [
            "No treatment required at this time",
            "Maintain regular watering schedule",
            "Apply balanced NPK fertilizer as per schedule",
        ],
        "prevention": [
            "Monitor plants weekly for early signs of disease",
            "Keep orchard floor clean — remove fallen leaves",
            "Maintain soil pH between 6.0 and 7.5",
            "Apply preventive neem oil spray once a month",
        ],
        "note": "✅ Plant appears healthy. Continue standard maintenance practices.",
    },
    "Guava_insect_bite": {
        "overview": (
            "Insect bite damage is caused by feeding insects such as fruit flies, "
            "thrips, or mealybugs. Irregular holes and necrotic patches are characteristic. "
            "Secondary fungal infections may follow."
        ),
        "symptoms": [
            "Irregular holes or torn edges on leaf surface",
            "Small necrotic (brown/black) spots at feeding sites",
            "Silvery or bronze discoloration from thrips feeding",
            "Sticky honeydew deposits indicating mealybug or aphid presence",
        ],
        "treatment": [
            "Spray Neem oil (5 ml/L water) — effective against soft-bodied insects",
            "Apply Imidacloprid 0.3 ml/L for severe infestations",
            "Use yellow sticky traps to monitor and reduce adult fly populations",
            "Introduce natural predators such as ladybugs if available",
        ],
        "prevention": [
            "Regularly inspect the undersides of leaves for egg clusters",
            "Remove and destroy heavily infested branches",
            "Avoid excessive nitrogen fertilization — promotes soft tissue",
            "Maintain orchard hygiene — remove fruit waste promptly",
        ],
        "note": "⚠️ RAG integration pending — advice based on curated agricultural knowledge base.",
    },
    "Guava_multiple": {
        "overview": (
            "Multiple concurrent diseases detected. This indicates a stressed plant with "
            "compromised immunity, likely under combined fungal and pest pressure. "
            "Immediate intervention is required."
        ),
        "symptoms": [
            "Multiple lesion types visible — fungal spots + insect damage",
            "Widespread leaf discoloration and necrosis",
            "Possible stem cankers or bark lesions",
            "Significant defoliation risk",
        ],
        "treatment": [
            "Apply broad-spectrum fungicide (Mancozeb 0.25%) immediately",
            "Follow with systemic insecticide (Chlorpyrifos 2 ml/L) after 3 days",
            "Remove all severely affected branches — sterilize pruning tools",
            "Foliar spray of micronutrients (Zinc + Boron) to boost immunity",
        ],
        "prevention": [
            "Conduct thorough disease scouting every 5–7 days",
            "Implement integrated pest management (IPM) program",
            "Improve drainage — waterlogged soil increases disease susceptibility",
            "Avoid plant stress — maintain consistent irrigation and nutrition",
        ],
        "note": "🚨 Multiple conditions detected. Consult a local agricultural extension officer for a combined treatment plan.",
    },
    "Guava_scorch": {
        "overview": (
            "Leaf scorch appears as browning of leaf margins and tips, caused by water "
            "stress, nutrient deficiency (especially potassium), or root damage. "
            "It is not infectious but indicates physiological stress."
        ),
        "symptoms": [
            "Brown, dry margins and leaf tips",
            "Yellowing of leaf tissue between veins (interveinal chlorosis)",
            "Brittle, papery texture at affected margins",
            "Symptoms progress from older leaves to younger leaves",
        ],
        "treatment": [
            "Improve irrigation — ensure consistent moisture, avoid drought stress",
            "Apply potassium sulfate (K2SO4) 2–3 kg per tree as soil drench",
            "Foliar spray of 0.5% potassium nitrate (KNO3) for quick uptake",
            "Check and treat root health — address compaction or root rot",
        ],
        "prevention": [
            "Mulch around the base to retain soil moisture",
            "Conduct soil test annually — maintain adequate K and Ca levels",
            "Avoid over-fertilization with nitrogen — causes K imbalance",
            "Provide shade during extreme heat if possible",
        ],
        "note": "⚠️ RAG integration pending — advice based on curated agricultural knowledge base.",
    },
    "Guava_yld": {
        "overview": (
            "Yellow Leaf Disease in guava is associated with phytoplasma infection or "
            "severe micronutrient deficiency (iron, magnesium, zinc). It causes progressive "
            "yellowing and is a significant yield threat if untreated."
        ),
        "symptoms": [
            "Uniform yellowing starting from leaf margins inward",
            "Veins may remain green initially (vein clearing)",
            "Stunted new growth and reduced leaf size",
            "Premature defoliation in advanced stages",
        ],
        "treatment": [
            "Soil application of ferrous sulfate (FeSO4) 50g per tree",
            "Foliar spray of 0.5% magnesium sulfate (MgSO4) every 2 weeks",
            "Apply chelated micronutrient mix (Fe + Zn + Mn) as foliar spray",
            "If phytoplasma suspected: apply Tetracycline antibiotic (consult specialist)",
        ],
        "prevention": [
            "Test soil pH — iron unavailability increases in alkaline soils (pH > 7.5)",
            "Apply organic compost annually to improve micronutrient availability",
            "Control leafhoppers — primary vector of phytoplasma",
            "Remove and destroy severely infected trees to prevent spread",
        ],
        "note": "🚨 If symptoms spread rapidly across the orchard, phytoplasma infection is suspected. Contact your local agriculture department.",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load Stage 2 model for inference.
    Cached so it only loads once per Streamlit session.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(STAGE2_PTH):
        st.error(f"Stage 2 checkpoint not found: {STAGE2_PTH}")
        st.info("Download stage2_best.pth from Drive → models/stage2/")
        st.stop()

    import timm

    # Checkpoint has head.weight [6,192] + head.bias [6] — plain Linear head
    # timm.create_model with num_classes=6 produces exactly this architecture
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=6)

    ckpt = torch.load(STAGE2_PTH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)  # all keys match now
    model.to(device)
    model.eval()

    return model, device


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Apply val transforms (no augmentation) and return (1, 3, 224, 224) tensor."""
    transform = get_transforms("val")
    tensor = transform(pil_image.convert("RGB"))
    return tensor.unsqueeze(0)  # add batch dim


@torch.no_grad()
def predict(model, tensor: torch.Tensor, device):
    """
    Run inference. Returns:
        top3: list of (class_name, confidence_float) sorted by confidence desc
        all_probs: full softmax probability array (6,)
    """
    tensor = tensor.to(device)
    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]  # shape (6,)

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(GUAVA_CLASSES[i], float(probs[i])) for i in top3_idx]

    return top3, probs


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION MAP
# ══════════════════════════════════════════════════════════════════════════════

def generate_attention_overlay(model, tensor: torch.Tensor, device, pil_image: Image.Image) -> Image.Image:
    """
    Extract attention from the last transformer block.
    CLS token attends to all 196 patches → reshape to 14×14 → upsample to 224×224.
    Overlay as a green-to-red heatmap on the original image.
    """
    tensor = tensor.to(device)
    attn_maps = get_attention_maps(model, tensor)

    # Last block attention: shape (1, num_heads, 197, 197)
    # DeiT-tiny has 3 heads, 197 tokens (1 CLS + 196 patches)
    last_attn = attn_maps[-1]  # (1, 3, 197, 197)

    # CLS token row: how much CLS attends to each patch
    cls_attn = last_attn[0, :, 0, 1:]  # (3, 196) — exclude CLS-to-CLS
    cls_attn = cls_attn.mean(dim=0)    # average across heads → (196,)
    cls_attn = cls_attn.cpu().numpy()

    # Normalize to [0, 1]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    # Reshape to 14×14 grid (sqrt(196) = 14 patches per side)
    attn_map = cls_attn.reshape(14, 14)

    # Upsample to 224×224 using PIL
    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (224, 224), Image.BILINEAR
    )
    attn_array = np.array(attn_pil) / 255.0

    # Apply colormap (jet: blue=low attention, red=high attention)
    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(attn_array)           # (224, 224, 4)
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil  = Image.fromarray(heatmap_rgb)

    # Resize original to 224×224 for clean overlay
    orig_resized = pil_image.convert("RGB").resize((224, 224), Image.LANCZOS)

    # Blend: 55% original, 45% heatmap
    overlay = Image.blend(orig_resized, heatmap_pil, alpha=0.45)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def confidence_bar_html(label: str, confidence: float, color: str, is_top: bool = False) -> str:
    """Render a styled confidence progress bar as HTML."""
    pct = confidence * 100
    bar_color = color if is_top else "#4a5568"
    bg = "rgba(255,255,255,0.08)" if is_top else "rgba(255,255,255,0.03)"
    border = f"1px solid {color}40" if is_top else "1px solid rgba(255,255,255,0.06)"
    font_weight = "700" if is_top else "400"

    return f"""
    <div style="
        background: {bg};
        border: {border};
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    ">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-family:'DM Sans',sans-serif; font-size:13px;
                         color:#e2e8f0; font-weight:{font_weight};">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:13px;
                         color:{color}; font-weight:700;">{pct:.1f}%</span>
        </div>
        <div style="background:rgba(255,255,255,0.08); border-radius:4px; height:6px; overflow:hidden;">
            <div style="width:{pct:.1f}%; height:100%; background:{bar_color};
                        border-radius:4px; transition:width 0.6s ease;"></div>
        </div>
    </div>
    """


def advice_card_html(icon: str, title: str, items: list, accent: str) -> str:
    """Render an advice card with bullet items."""
    items_html = "".join(
        f'<li style="margin-bottom:6px; color:#cbd5e0; font-size:13px; line-height:1.5;">{item}</li>'
        for item in items
    )
    return f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid {accent}30;
        border-left: 3px solid {accent};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    ">
        <div style="font-family:'DM Sans',sans-serif; font-size:14px;
                    font-weight:700; color:{accent}; margin-bottom:10px;">
            {icon} {title}
        </div>
        <ul style="margin:0; padding-left:18px;">
            {items_html}
        </ul>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GuavaScan — Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global reset ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0d1117;
}
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

/* ── Hide Streamlit chrome ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Upload widget ────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(52, 211, 153, 0.4) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(52, 211, 153, 0.8) !important;
}
[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons ──────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #34d399, #059669) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Dividers ─────────────────────────────────────── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.5rem 0 !important;
}

/* ── Spinner ──────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #34d399 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(52,211,153,0.12) 0%, rgba(5,150,105,0.06) 100%);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute; top: -30px; right: -30px;
        width: 160px; height: 160px;
        background: radial-gradient(circle, rgba(52,211,153,0.15) 0%, transparent 70%);
        border-radius: 50%;
    "></div>
    <div style="display: flex; align-items: center; gap: 16px;">
        <div style="font-size: 48px; line-height:1;">🌿</div>
        <div>
            <h1 style="
                font-family: 'DM Serif Display', serif;
                font-size: 2.2rem;
                color: #f0fdf4;
                margin: 0 0 4px 0;
                letter-spacing: -0.5px;
            ">GuavaScan</h1>
            <p style="
                font-family: 'DM Sans', sans-serif;
                font-size: 14px;
                color: #6ee7b7;
                margin: 0;
                letter-spacing: 0.5px;
            ">LEAF DISEASE DETECTION · DeiT-tiny · Two-Stage Transfer Learning · ViT + RAG</p>
        </div>
    </div>
    <div style="
        display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap;
    ">
        <span style="background:rgba(52,211,153,0.15); border:1px solid rgba(52,211,153,0.3);
                     border-radius:6px; padding:4px 10px; font-size:11px; color:#6ee7b7;
                     font-family:'JetBrains Mono',monospace;">Stage 2 · 6 Guava Classes</span>
        <span style="background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3);
                     border-radius:6px; padding:4px 10px; font-size:11px; color:#a5b4fc;
                     font-family:'JetBrains Mono',monospace;">99.80% → 100% Accuracy</span>
        <span style="background:rgba(251,191,36,0.12); border:1px solid rgba(251,191,36,0.25);
                     border-radius:6px; padding:4px 10px; font-size:11px; color:#fcd34d;
                     font-family:'JetBrains Mono',monospace;">RAG Integration · Pending</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL (cached)
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading GuavaScan model..."):
    model, device = load_model()

device_label = "GPU (CUDA)" if str(device) == "cuda" else "CPU"
st.markdown(f"""
<div style="text-align:right; font-family:'JetBrains Mono',monospace;
            font-size:11px; color:#4ade80; margin-top:-16px; margin-bottom:16px;">
    ● Model loaded · {device_label}
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — two columns
# ══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif; font-size:11px; letter-spacing:1.5px;
                color:#6ee7b7; font-weight:600; margin-bottom:12px;">
        📤 UPLOAD LEAF IMAGE
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a guava leaf photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    show_attention = False
    pil_image = None

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")

        # Show controls row
        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            show_attention = st.toggle("🔥 Show Attention Map", value=False)
        with ctrl_col2:
            run_btn = st.button("🔍 Analyse Leaf", use_container_width=True)

        # Image display
        if show_attention:
            with st.spinner("Generating attention map..."):
                tensor = preprocess(pil_image)
                overlay = generate_attention_overlay(model, tensor, device, pil_image)
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown('<p style="font-size:11px; color:#94a3b8; text-align:center; margin-bottom:4px;">Original</p>', unsafe_allow_html=True)
                st.image(pil_image.resize((224, 224)), use_container_width=True)
            with img_col2:
                st.markdown('<p style="font-size:11px; color:#94a3b8; text-align:center; margin-bottom:4px;">Attention Heatmap</p>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

            st.markdown("""
            <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                        border-radius:8px; padding:10px 14px; margin-top:8px;">
                <span style="font-size:11px; color:#a5b4fc; font-family:'DM Sans',sans-serif;">
                    🔬 <b>Explainability:</b> Red/warm regions = areas the model focused on for its prediction.
                    A good model should highlight diseased leaf regions, not background.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.image(pil_image, use_container_width=True, caption="Uploaded leaf image")

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 16px;
            height: 260px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
        ">
            <div style="font-size: 48px; opacity:0.3;">🍃</div>
            <p style="color:#4a5568; font-size:13px; margin:0; font-family:'DM Sans',sans-serif;">
                Upload a guava leaf photo to begin diagnosis
            </p>
            <p style="color:#374151; font-size:11px; margin:0; font-family:'JetBrains Mono',monospace;">
                JPG · PNG · WEBP · max 200MB
            </p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results
# ══════════════════════════════════════════════════════════════════════════════

with col_right:
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif; font-size:11px; letter-spacing:1.5px;
                color:#6ee7b7; font-weight:600; margin-bottom:12px;">
        📊 DISEASE ANALYSIS
    </div>
    """, unsafe_allow_html=True)

    if pil_image is None:
        # Empty state
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 48px 24px;
            text-align: center;
        ">
            <div style="font-size:36px; opacity:0.2; margin-bottom:12px;">🔬</div>
            <p style="color:#374151; font-size:13px; font-family:'DM Sans',sans-serif; margin:0;">
                Results will appear here after upload
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Run prediction
        with st.spinner("Analysing leaf..."):
            tensor = preprocess(pil_image)
            top3, all_probs = predict(model, tensor, device)

        top_class, top_conf = top3[0]
        display_name = DISPLAY_NAMES[top_class]
        severity_label, severity_color = SEVERITY[top_class]

        # ── Primary result card ──────────────────────────────
        is_healthy = top_class == "Guava_healthy"
        card_accent = "#34d399" if is_healthy else severity_color

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {card_accent}18 0%, {card_accent}08 100%);
            border: 1px solid {card_accent}40;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-family:'DM Sans',sans-serif; font-size:11px;
                                letter-spacing:1.2px; color:#94a3b8; margin-bottom:6px;">
                        DETECTED DISEASE
                    </div>
                    <div style="font-family:'DM Serif Display',serif; font-size:1.8rem;
                                color:#f8fafc; line-height:1.1;">
                        {display_name}
                    </div>
                </div>
                <div style="
                    background: {severity_color}25;
                    border: 1px solid {severity_color}60;
                    border-radius: 8px;
                    padding: 6px 14px;
                    text-align: center;
                ">
                    <div style="font-size:10px; color:{severity_color}; letter-spacing:1px;
                                font-family:'JetBrains Mono',monospace;">SEVERITY</div>
                    <div style="font-size:15px; color:{severity_color}; font-weight:700;
                                font-family:'DM Sans',sans-serif;">{severity_label}</div>
                </div>
            </div>
            <div style="margin-top:14px;">
                <div style="font-size:11px; color:#94a3b8; margin-bottom:6px;
                            font-family:'JetBrains Mono',monospace;">CONFIDENCE</div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="flex:1; background:rgba(255,255,255,0.08); border-radius:6px;
                                height:10px; overflow:hidden;">
                        <div style="width:{top_conf*100:.1f}%; height:100%;
                                    background: linear-gradient(90deg, {card_accent}, {card_accent}cc);
                                    border-radius:6px;"></div>
                    </div>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:18px;
                                 color:{card_accent}; font-weight:700; min-width:60px;">
                        {top_conf*100:.1f}%
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-3 breakdown ──────────────────────────────────
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:11px; letter-spacing:1.2px;
                    color:#94a3b8; font-weight:600; margin-bottom:10px;">
            TOP 3 PREDICTIONS
        </div>
        """, unsafe_allow_html=True)

        colors_top3 = [card_accent, "#60a5fa", "#a78bfa"]
        for i, (cls, conf) in enumerate(top3):
            bar_html = confidence_bar_html(
                DISPLAY_NAMES[cls], conf,
                colors_top3[i], is_top=(i == 0)
            )
            st.markdown(bar_html, unsafe_allow_html=True)

        # ── Model info footer ────────────────────────────────
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 8px;
            display: flex;
            gap: 20px;
        ">
            <span style="font-size:11px; color:#4b5563; font-family:'JetBrains Mono',monospace;">
                Model: DeiT-tiny Stage 2
            </span>
            <span style="font-size:11px; color:#4b5563; font-family:'JetBrains Mono',monospace;">
                Device: {device_label}
            </span>
            <span style="font-size:11px; color:#4b5563; font-family:'JetBrains Mono',monospace;">
                Classes: 6 Guava
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TREATMENT ADVICE — full width below
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
    <div>
        <div style="font-family:'DM Sans',sans-serif; font-size:11px; letter-spacing:1.5px;
                    color:#fcd34d; font-weight:600;">
            🤖 AI TREATMENT ADVICE
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#6b7280; margin-top:2px;">
            Knowledge base · RAG integration pending — full semantic retrieval coming in Phase 3
        </div>
    </div>
    <div style="
        background: rgba(251,191,36,0.1);
        border: 1px solid rgba(251,191,36,0.25);
        border-radius: 6px;
        padding: 4px 12px;
        font-size:11px;
        color:#fcd34d;
        font-family:'JetBrains Mono',monospace;
        white-space: nowrap;
    ">⚠ Hardcoded · Pre-RAG</div>
</div>
""", unsafe_allow_html=True)

if pil_image is None:
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.02);
        border: 1px dashed rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 36px;
        text-align: center;
        color: #374151;
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
    ">
        Upload and analyse a leaf image to see treatment recommendations
    </div>
    """, unsafe_allow_html=True)

else:
    advice = DISEASE_ADVICE[top_class]

    # Overview
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: #cbd5e0;
        line-height: 1.6;
    ">
        {advice['overview']}
    </div>
    """, unsafe_allow_html=True)

    # Three-column card layout
    adv_col1, adv_col2, adv_col3 = st.columns(3)

    with adv_col1:
        st.markdown(advice_card_html(
            "🔍", "Symptoms to Watch",
            advice["symptoms"], "#f87171"
        ), unsafe_allow_html=True)

    with adv_col2:
        st.markdown(advice_card_html(
            "💊", "Treatment Steps",
            advice["treatment"], "#34d399"
        ), unsafe_allow_html=True)

    with adv_col3:
        st.markdown(advice_card_html(
            "🛡️", "Prevention Measures",
            advice["prevention"], "#60a5fa"
        ), unsafe_allow_html=True)

    # Note / disclaimer
    st.markdown(f"""
    <div style="
        background: rgba(251,191,36,0.06);
        border: 1px solid rgba(251,191,36,0.2);
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: #fcd34d;
        margin-top: 4px;
    ">
        {advice['note']}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    text-align: center;
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #374151;
">
    GuavaScan · Phase 2 Presentation · DeiT-tiny + Two-Stage Transfer Learning
    · <span style="color:#34d399;">Stage 1: 99.80% (44 classes)</span>
    · <span style="color:#34d399;">Stage 2: 100% (6 guava classes)</span>
    · RAG integration planned for Phase 3
</div>
""", unsafe_allow_html=True)