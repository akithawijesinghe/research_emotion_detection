# ===========================================================
# 🎭 Emotion Detection App — Fine-Tuned BERT (Research Version)
# ===========================================================

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import numpy as np
import pandas as pd
import altair as alt

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "models"
CONFIDENCE_THRESHOLD = 0.5   # research-chosen threshold
MIN_WORDS = 3               # minimum words for valid inference

# -----------------------------
# Load BERT Model
# -----------------------------
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    with open(f"{MODEL_PATH}/id2label.json", "r") as f:
        id2label = json.load(f)
    return tokenizer, model, id2label

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🎭",
    layout="centered"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    textarea {
        border: 2px solid #4CAF50 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
    }
    textarea:focus {
        border-color: #00E676 !important;
        box-shadow: 0 0 6px rgba(0, 230, 118, 0.6) !important;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.3rem !important;
    }
    div.stButton > button {
        height: 36px;
        padding: 0.2rem 0.6rem;
        font-size: 0.82rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .quick-examples {
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("🎭 Emotion Detection from Text (BERT)")
st.write("Fine-tuned BERT model trained for multi-emotion classification.")

# -----------------------------
# Model Info (Expandable)
# -----------------------------
with st.expander("ℹ️ Model Information"):
    st.markdown("""
    **Model:** Fine-tuned BERT-base  
    **Emotion Classes:** anger, anxiety, fear, grief, sadness  
    **Training Data:** 12,742 labeled samples  
    **Evaluation Metric:** Macro F1-score  
    **Inference Strategy:**  
    • Minimum context validation  
    • Confidence-based rejection for out-of-scope inputs  
    """)

# -----------------------------
# Load Model
# -----------------------------
tokenizer, model, id2label = load_bert_model()

# -----------------------------
# Initialize Session State
# -----------------------------
if 'selected_text' not in st.session_state:
    st.session_state.selected_text = ""

# -----------------------------
# Predefined Examples
# -----------------------------
st.markdown('<div class="quick-examples">💭 <strong>Quick Examples</strong></div>', unsafe_allow_html=True)

predefined_sentences = [
    "I’m still scared",
    "Hearts are broken",
    "No mercy anymore"
]

# Create horizontal layout with compact buttons
cols = st.columns(3)
for idx, sentence in enumerate(predefined_sentences):
    with cols[idx]:
        if st.button(
            sentence,
            key=f"example_{idx}",
            use_container_width=True
        ):
            st.session_state.selected_text = sentence
            st.rerun()

# -----------------------------
# Text Input
# -----------------------------
text_input = st.text_area(
    "Enter your comment:",
    value=st.session_state.selected_text,
    height=120
)

st.caption("⚠️ Short or ambiguous sentences may reduce prediction accuracy.")

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict Emotion"):
    if not text_input.strip():
        st.warning("Please enter some text before predicting.")

    else:
        # -------- Minimum Word Validation --------
        word_count = len(text_input.strip().split())
        if word_count < MIN_WORDS:
            st.info(
                "ℹ️ Please enter at least **three words** to provide sufficient "
                "linguistic context for emotion analysis."
            )

        else:
            # -------- Tokenization --------
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

            max_prob = float(np.max(probs))
            pred_id = int(np.argmax(probs))
            emotion = id2label[str(pred_id)]

            # -------- Out-of-Scope Handling --------
            if max_prob < CONFIDENCE_THRESHOLD:
                st.warning(
                    "⚠️ **Emotion not confidently detected**\n\n"
                    "The input does not strongly correspond to any of the trained "
                    "emotion categories."
                )

            # -------- Confident Prediction --------
            else:
                st.success(
                    f"Predicted Emotion: **{emotion.upper()}** 🎯 "
                    f"(Confidence: {max_prob:.2f})"
                )

                # Prepare chart data
                df = pd.DataFrame({
                    "Emotion": [id2label[str(i)] for i in range(len(probs))],
                    "Probability": probs
                })

                # -------- FIXED BAR CHART (HORIZONTAL LABELS) --------
                chart = (
                    alt.Chart(df)
                    .mark_bar(size=32, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X(
                            "Emotion:N",
                            sort=None,
                            title="Emotion",
                            axis=alt.Axis(labelAngle=0, labelPadding=10)
                        ),
                        y=alt.Y(
                            "Probability:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            title="Probability"
                        ),
                        color=alt.condition(
                            alt.datum.Probability == max_prob,
                            alt.value("#00E676"),
                            alt.value("#90CAF9")
                        ),
                        tooltip=[
                            "Emotion",
                            alt.Tooltip("Probability:Q", format=".3f")
                        ]
                    )
                    .properties(height=300)
                )

                st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Academic Disclaimer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ This system is designed for crisis-related emotional expressions. "
    "Neutral or positive text may fall outside the trained emotion categories."
)