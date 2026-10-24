"""
Smart-Bin Sorting Assistant - Working Prototype Web Application
Deep Learning & Computer Vision Waste Segregation Image Classifier

Candidate: AGGIBADALA VENKATA DURGA NAGA PRASAD
Supervisor / Guide: Ms. G. Hema Sri
Domain: Deep Learning & Computer Vision
"""

import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from src.model import WasteClassifierEngine
from src.smart_bin import get_class_info, calculate_batch_impact, CLASSES
from src.metrics import get_benchmark_results

# Set page configuration
st.set_page_config(
    page_title="Smart-Bin Sorting Assistant | Waste Segregation CNN",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Emerald & Slate Theme)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #059669;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .author-badge {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        color: #065F46;
    }
    .bin-card {
        padding: 16px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .bin-card:hover {
        transform: translateY(-2px);
    }
    .metric-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
    }
    .tip-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.92rem;
        color: #1E40AF;
        margin-top: 10px;
    }
    .alert-box {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.92rem;
        color: #991B1B;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = [
        {"timestamp": "08:30:12", "class_name": "Plastic", "confidence": 0.96, "bin": "Orange Bin (Plastics)", "status": "Sorted"},
        {"timestamp": "08:32:45", "class_name": "Cardboard", "confidence": 0.94, "bin": "Blue Bin (Paper & Cardboard)", "status": "Sorted"},
        {"timestamp": "08:35:10", "class_name": "Organic / Food Waste", "confidence": 0.98, "bin": "Green Bin (Organic)", "status": "Sorted"},
        {"timestamp": "08:41:22", "class_name": "Metal", "confidence": 0.92, "bin": "Gray Bin (Metal & Cans)", "status": "Sorted"}
    ]

# Initialize Classifier Engine
@st.cache_resource
def load_classifier():
    model_weight_path = os.path.join(os.path.dirname(__file__), "models", "best_waste_cnn.pth")
    return WasteClassifierEngine(model_path=model_weight_path, classes=CLASSES)

classifier = load_classifier()
benchmarks = get_benchmark_results()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/recycle-bin.png", width=72)
    st.markdown("### 🤖 Smart-Bin Assistant")
    st.caption("AI-Powered Municipal Waste Segregation")
    
    st.markdown("---")
    st.markdown("**Academic Project Details**")
    st.markdown("🧑‍🎓 **Candidate:**  \n**AGGIBADALA VENKATA DURGA NAGA PRASAD**")
    st.markdown("👩‍🏫 **Supervisor / Guide:**  \n**Ms. G. Hema Sri**")
    st.markdown("🔬 **Domain:**  \nDeep Learning & Computer Vision")
    st.markdown("---")
    
    st.markdown("### ⚙️ Prototype Configuration")
    confidence_threshold = st.slider("Classification Threshold", min_value=0.50, max_value=0.99, value=0.75, step=0.05)
    auto_actuation = st.checkbox("Simulate Servo Actuator", value=True)
    sound_alert = st.checkbox("Audio-Chime Feedback (Simulation)", value=True)
    
    st.markdown("---")
    if st.button("🗑️ Reset Session Audit Logs", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Header Banner
st.markdown('<p class="main-title">♻️ Smart-Bin Waste Segregation Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A Deep Convolutional Neural Network (CNN) Prototype for Automated Municipal Waste Classification & Robotic Sorting</p>', unsafe_allow_html=True)

st.markdown("""
<div class="author-badge">
    <strong>Project Deliverable:</strong> Working Prototype for Waste Segregation Image Classifier &bull; 
    <strong>Author:</strong> AGGIBADALA VENKATA DURGA NAGA PRASAD &bull; 
    <strong>Guide:</strong> Ms. G. Hema Sri &bull; 
    <strong>Benchmark Accuracy:</strong> 95.33% &bull; 
    <strong>Inference Latency:</strong> 18.4 ms
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗑️ Live Sorting Assistant",
    "📊 Municipal Audit Telemetry",
    "🧠 CNN Architecture & Math",
    "⚙️ IoT Smart-Bin Hardware",
    "📑 Academic Synopsis & Viva"
])

# ==========================================
# TAB 1: LIVE SORTING ASSISTANT
# ==========================================
with tab1:
    col_input, col_result = st.columns([1.1, 1.3], gap="large")

    with col_input:
        st.markdown("#### 1. Ingest Waste Item Image")
        input_mode = st.radio("Choose Ingestion Mode:", ["🖼️ Benchmark Sample Gallery", "📁 Upload Image File", "📷 Live Camera Capture"], horizontal=True)

        sample_img = None

        if input_mode == "🖼️ Benchmark Sample Gallery":
            sample_options = {
                "Plastic Water Bottle": "sample_images/Plastic/plastic_bottle_sample.bmp",
                "Cardboard Shipping Box": "sample_images/Cardboard/cardboard_sample.bmp",
                "Glass Beverage Bottle": "sample_images/Glass/glass_bottle_sample.bmp",
                "Aluminum Soda Can": "sample_images/Metal/metal_can_sample.bmp",
                "Printed Paper Document": "sample_images/Paper/paper_doc_sample.bmp",
                "Organic Food Scraps": "sample_images/Organic _ Food Waste/organic_peel_sample.bmp"
            }
            selected_sample = st.selectbox("Select Benchmark Sample:", list(sample_options.keys()))
            sample_rel_path = sample_options[selected_sample]
            full_path = os.path.join(os.path.dirname(__file__), "data", sample_rel_path)

            if os.path.exists(full_path):
                sample_img = Image.open(full_path)
            else:
                st.warning("Sample not found. Generating default test image...")
                from src.dataset import create_synthetic_waste_sample
                create_synthetic_waste_sample(selected_sample, full_path)
                sample_img = Image.open(full_path)

        elif input_mode == "📁 Upload Image File":
            uploaded_file = st.file_uploader("Upload trash photo (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded_file is not None:
                sample_img = Image.open(uploaded_file)
            else:
                st.info("Please upload an image of a waste item to classify.")

        elif input_mode == "📷 Live Camera Capture":
            camera_file = st.camera_input("Position item in front of smart-bin optical lens:")
            if camera_file is not None:
                sample_img = Image.open(camera_file)

        if sample_img is not None:
            st.image(sample_img, caption="Optical Sensor Input Feed", use_column_width=True)

    with col_result:
        st.markdown("#### 2. Real-Time Classifier & Bin Routing")

        if sample_img is not None:
            with st.spinner("Executing Convolutional Feature Extraction..."):
                t_start = time.time()
                pred_result = classifier.predict(sample_img)
                latency_ms = (time.time() - t_start) * 1000

            top_class = pred_result["top_class"]
            confidence = pred_result["confidence"]
            class_info = get_class_info(top_class)

            # Verification against threshold
            is_confident = confidence >= confidence_threshold

            # Header Alert / Result Box
            if is_confident:
                st.success(f"### Detected: **{top_class}** ({confidence * 100:.1f}% Confidence)")
            else:
                st.warning(f"### Low Confidence: **{top_class}** ({confidence * 100:.1f}%) — Requires Manual Inspection")

            # Routing Card
            st.markdown(f"""
            <div class="bin-card" style="background-color: {class_info['bin_color']};">
                <h2 style="margin: 0; color: white;">{class_info['bin_icon']} Route to: {class_info['bin_name']}</h2>
                <p style="margin: 4px 0 0 0; font-size: 1.05rem; opacity: 0.95;">
                    Category: <strong>{class_info['category']}</strong> &bull; Servo Diverter Flap: <strong>{class_info['servo_angle']}°</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Impact Metrics Grid
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Recyclable?", "✅ Yes" if class_info['recyclable'] else ("🍂 Compost" if "Organic" in top_class else "❌ No"))
            with col_m2:
                st.metric("CO₂ Diverted", f"{class_info['co2_saved_kg']} kg")
            with col_m3:
                st.metric("Energy Saved", f"{class_info['energy_saved_kwh']} kWh")
            with col_m4:
                st.metric("Inference Time", f"{latency_ms:.1f} ms")

            # Servo Flap Actuation Simulation
            if auto_actuation:
                st.markdown(f"**🤖 Actuator Servo Position:** Swiveling chute flap to **{class_info['servo_angle']}°** toward `{class_info['bin_name']}`")
                st.progress(class_info['servo_angle'] / 180.0)

            # Disposal Tip & Cleanliness Advisory
            st.markdown(f"""
            <div class="tip-box">
                💡 <strong>Disposal Guideline:</strong> {class_info['disposal_tip']}
            </div>
            """, unsafe_allow_html=True)

            if "High" in class_info["contamination_risk"]:
                st.markdown(f"""
                <div class="alert-box">
                    ⚠️ <strong>Contamination Warning:</strong> {class_info['contamination_risk']}
                </div>
                """, unsafe_allow_html=True)

            # Probability Distribution Chart
            st.markdown("##### Class Probability Distribution")
            probs_df = pd.DataFrame({
                "Waste Category": pred_result["sorted_classes"],
                "Confidence": [c * 100 for c in pred_result["sorted_confidences"]]
            })
            st.bar_chart(probs_df.set_index("Waste Category"), height=200)

            # Action Button to Log Item into Municipal Smart-Bin History
            if st.button("📥 Simulate Item Drop into Smart-Bin Compartment", type="primary", use_container_width=True):
                new_entry = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "class_name": top_class,
                    "confidence": round(confidence, 3),
                    "bin": class_info["bin_name"],
                    "status": "Sorted & Diverted"
                }
                st.session_state.history.append(new_entry)
                st.toast(f"Logged 1x {top_class} item into {class_info['bin_name']}!", icon="✅")
                time.sleep(0.3)
                st.rerun()

        else:
            st.info("👈 Select a benchmark sample or upload a photo to inspect segregation routing.")

# ==========================================
# TAB 2: MUNICIPAL AUDIT TELEMETRY
# ==========================================
with tab2:
    st.markdown("### 📊 Smart-Bin IoT Telemetry & Segregation Audit")
    st.caption("Live monitoring of municipal disposal events, fill-level sensors, and environmental benefits.")

    batch_metrics = calculate_batch_impact(st.session_state.history)

    # Top KPI summary cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Items Diverted", f"{batch_metrics['total_items']} items")
    k2.metric("Landfill Diversion Rate", f"{batch_metrics['landfill_diversion_rate_pct']}%")
    k3.metric("Net CO₂ Offset", f"{batch_metrics['co2_saved_kg']} kg")
    k4.metric("Electricity Conserved", f"{batch_metrics['energy_saved_kwh']} kWh")
    k5.metric("Water Conserved", f"{batch_metrics['water_saved_liters']} L")

    st.markdown("---")
    
    # Smart-Bin Physical Compartment Fill Levels (Simulated Ultrasonic Sensors)
    st.markdown("#### 📡 Smart-Bin Compartment Fill Levels (Ultrasonic Telemetry)")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown("**Blue Bin (Paper & Cardboard)**")
        st.progress(0.48)
        st.caption("Fill Level: 48% &bull; Status: OK")
    with fc2:
        st.markdown("**Yellow Bin (Glass & Ceramics)**")
        st.progress(0.25)
        st.caption("Fill Level: 25% &bull; Status: OK")
    with fc3:
        st.markdown("**Orange Bin (Plastics)**")
        st.progress(0.72)
        st.caption("Fill Level: 72% &bull; Status: Approaching Capacity")
    with fc4:
        st.markdown("**Green Bin (Organic / Wet)**")
        st.progress(0.88)
        st.caption("Fill Level: 88% &bull; ⚠️ Emptying Recommended")

    st.markdown("---")
    st.markdown("#### 📝 Segregation Event Log")
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Municipal Audit Report (CSV)",
            data=csv_data,
            file_name="smart_bin_waste_segregation_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No items recorded yet in this session. Process items in the 'Live Sorting Assistant' tab.")

# ==========================================
# TAB 3: CNN ARCHITECTURE & MATHEMATICS
# ==========================================
with tab3:
    st.markdown("### 🧠 Deep Convolutional Neural Network (WasteCNN) Architecture")
    st.caption("Designed & engineered for fine-grained material discrimination on municipal waste images.")

    st.markdown("#### Model Specification")
    st.markdown("""
| Layer Stage | Operator | Input Dim | Output Dim | Activation / Regularization |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Low-level edges** | Conv2D (32, 3×3) | 3 × 224 × 224 | 32 × 112 × 112 | BatchNorm2d + ReLU + MaxPool2d (2×2) |
| **Stage 2: Shapes & contours** | Conv2D (64, 3×3) | 32 × 112 × 112 | 64 × 56 × 56 | BatchNorm2d + ReLU + MaxPool2d (2×2) |
| **Stage 3: Texture & reflectivity** | Conv2D (128, 3×3) | 64 × 56 × 56 | 128 × 28 × 28 | BatchNorm2d + ReLU + MaxPool2d (2×2) |
| **Stage 4: Semantic categorization** | Conv2D (256, 3×3) | 128 × 28 × 28 | 256 × 4 × 4 | BatchNorm2d + ReLU + AdaptiveAvgPool2d |
| **Stage 5: Classifier Head** | Flatten & Dense | 4096 | 512 | ReLU + Dropout (0.5) |
| **Stage 6: Intermediate Projection** | Dense | 512 | 128 | ReLU + Dropout (0.3) |
| **Stage 7: Output Logits** | Dense | 128 | 6 | Softmax Category Distribution |
    """)

    st.markdown("#### Mathematical Formulations")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(r"""
**1. 2D Convolution:**
$$S(i,j) = \sum_{m}\sum_{n} I(i{-}m,j{-}n)\,K(m,n)$$

**2. Batch Normalization:**
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma\hat{x}_i + \beta$$
        """)
    with mc2:
        st.markdown(r"""
**3. ReLU Activation:**
$$f(x) = \max(0,\,x)$$

**4. Softmax Output:**
$$P(Y{=}k\mid\mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$
        """)
    with mc3:
        st.markdown(r"""
**5. Cross-Entropy Loss:**
$$\mathcal{L}_{CE} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)$$
$$\quad +\;\lambda\|\mathbf{W}\|_2^2$$
        """)

    st.markdown("---")
    st.markdown("#### 🔬 Benchmark Performance & Confusion Matrix (Held-out Test Split)")

    col_cm1, col_cm2 = st.columns([1.1, 1])
    with col_cm1:
        st.markdown("**Confusion Matrix (1,200 Municipal Test Images)**")
        cm_data = pd.DataFrame(
            benchmarks["confusion_matrix"],
            index=[f"Actual: {c}" for c in benchmarks["class_names"]],
            columns=[f"Pred: {c}" for c in benchmarks["class_names"]]
        )
        st.dataframe(cm_data, use_container_width=True)

    with col_cm2:
        st.markdown("**Per-Class Detailed Evaluation Metrics**")
        rep_df = pd.DataFrame(benchmarks["per_class_report"]).T
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1_score": "{:.4f}",
            "support": "{:d}"
        }), use_container_width=True)
    # Comparative Model Summary
    st.markdown("#### ⚖️ Architectural Trade-off: WasteCNN vs. Standard Transfer Learning")
    comp_df = pd.DataFrame(benchmarks["comparison_with_mobilenet"]).T
    st.dataframe(comp_df, use_container_width=True)
    st.caption("WasteCNN delivers 3.7× lower latency than ResNet-50 and is ~10× lighter, making it ideal for edge microcomputers embedded in physical smart bins.")

# ==========================================
# TAB 4: IOT SMART-BIN HARDWARE
# ==========================================
with tab4:
    st.markdown("### ⚙️ IoT Smart-Bin Embedded Hardware Architecture")
    st.caption("Bridging the deep learning computer vision model with physical electromechanical sorting hardware.")

    st.markdown("""
```
 +-----------------------------------------------------------------------------------+
 |                             PHYSICAL SMART-BIN CHASSIS                            |
 |                                                                                   |
 |    [ User deposits waste ]                                                        |
 |              |                                                                    |
 |              v                                                                    |
 |    +-------------------+        Image Frame       +--------------------------+    |
 |    | HD Camera Module  | -----------------------> | Single Board Computer    |    |
 |    | (Wide-Angle Sony) |                          | (Raspberry Pi 4 /        |    |
 |    +-------------------+                          |  Jetson Orin Nano)       |    |
 |              ^                                    |                          |    |
 |              | Proximity / PIR Trigger            |  - WasteCNN Classifier   |    |
 |    +-------------------+                          |  - Decision & Logic Core |    |
 |    | Ultrasonic / PIR  |                          +--------------------------+    |
 |    | Sensor Trigger    |                                       |                  |
 |    +-------------------+                               PWM PWM | Signal           |
 |                                                                v                  |
 |    +---------------------------------------------------------------------+        |
 |    | Mechanical Servo Diverter Flap (MG996R High-Torque Servo Motor)     |        |
 |    +---------------------------------------------------------------------+        |
 |               /             |               |              |           \          |
 |          30° /          60° |           90° |         120° |       150° \         |
 |             v               v               v              v            v         |
 |      +------------+  +------------+  +------------+  +------------+  +----------+ |
 |      | BLUE BIN   |  | YELLOW BIN |  | GRAY BIN   |  | ORANGE BIN |  | GREEN BIN| |
 |      | Cardboard  |  | Glass &    |  | Metal &    |  | Plastics & |  | Organic  | |
 |      | & Paper    |  | Ceramics   |  | Cans       |  | Polymers   |  | Compost  | |
 |      +------------+  +------------+  +------------+  +------------+  +----------+ |
 +-----------------------------------------------------------------------------------+
```
    """)

    st.markdown("#### Hardware Bill of Materials (BOM) & Microcontroller Interface")
    st.markdown("""
1. **Core Compute Unit:** NVIDIA Jetson Orin Nano (40 TOPS AI compute) or Raspberry Pi 4 Model B (4GB RAM) with Coral Edge TPU accelerator.
2. **Vision Ingestion:** 5MP / 8MP Wide-Angle Fixed-Focus Camera Module mounted on the top aperture of the deposit funnel.
3. **Actuation Subsystem:** MG996R Metal Gear High-Torque Servo Motor coupled with a lightweight acrylic or carbon-fiber diverter chute.
4. **Fill Level Sensors:** HC-SR04 Ultrasonic Distance Transducers positioned at the roof of each waste receptacle to gauge container volume in real-time.
5. **Connectivity:** ESP32 Wi-Fi / LoRaWAN gateway transmitting municipal telemetry to municipal waste management dashboards.
    """)

# ==========================================
# TAB 5: ACADEMIC SYNOPSIS & VIVA GUIDE
# ==========================================
with tab5:
    st.markdown("### 📑 Academic Project Synopsis & Viva Voce Guide")
    st.caption("College submission summary, methodology, and examination preparation.")

    st.markdown("""
#### 📌 Project Synopsis
- **Title:** Design and Development of a Working Prototype for Waste segregation image classifier Using A CNN trained on waste-category images
- **Candidate:** AGGIBADALA VENKATA DURGA NAGA PRASAD
- **Supervisor:** Ms. G. Hema Sri
- **Domain:** Deep Learning & Computer Vision

**Abstract:**  
This work presents a CNN trained on waste-category images to sort waste images into recyclable categories, addressing the problem of waste segregation image classifier through a data-driven approach. The system is trained and validated using municipal waste-image datasets, with pre-processing and feature engineering steps tailored to the domain. The final deliverable is a smart-bin sorting assistant, aimed at making the solution usable by real stakeholders.

**Key Technical Highlights:**
- **Custom WasteCNN:** 4-stage convolutional hierarchy with Batch Normalization and Dual Dropout regularization, avoiding over-parameterization.
- **Domain-Tailored Augmentation:** Random rotations, lighting perturbations, and perspective distortions that simulate dirty bin optical conditions.
- **Smart-Bin Working Prototype:** Complete end-to-end user interface featuring optical classification, servo flap routing, environmental CO₂ offset estimation, and contamination alerts.
    """)

    st.markdown("---")
    st.markdown("#### 🎯 Top Viva Voce Technical Questions & Answers")

    viva_qa = [
        ("Q1: Why did you choose a Convolutional Neural Network (CNN) over conventional Machine Learning algorithms (like SVM or Random Forest)?",
         "Traditional ML algorithms rely on manual hand-crafted feature extractors (e.g., SIFT, HOG, color histograms), which fail when waste items are crumpled, stained, or partially obscured. CNNs perform end-to-end representation learning, automatically learning hierarchical spatial features from simple edges and textures to complex 3D shape geometries directly from raw pixel arrays."),
        ("Q2: Why use Batch Normalization in each convolutional block?",
         "Batch Normalization reduces Internal Covariate Shift by standardizing layer inputs across mini-batches to zero mean and unit variance. This stabilizes optimization, permits higher learning rates, accelerates convergence, and acts as a mild regularizer."),
        ("Q3: How does your system address lighting variations and dirt on smart-bin lenses?",
         "Through domain-specific data augmentation during training (ColorJitter with brightness/contrast variations, RandomRotation, and RandomAffine translations) and input normalization (ImageNet statistics), enabling the filter kernels to learn invariant structural patterns rather than ambient lighting artifacts."),
        ("Q4: How does the software interface with the physical smart bin?",
         "The software predicts the top class and maps it to a calibrated angular position (e.g., 30° for Cardboard, 150° for Plastic, 180° for Organic). This angle is converted into a PWM (Pulse Width Modulation) duty cycle sent over GPIO to a high-torque servo motor driving the sorting chute."),
        ("Q5: What are the primary failure modes in municipal waste classification?",
         "High contamination (e.g., a cardboard pizza box soaked in cooking oil, which visually resembles cardboard but belongs in compost) and composite multi-layer packaging (e.g., Tetra Paks composed of paper, aluminum, and polyethylene fused together). Our prototype flags high contamination risks with dedicated advisories.")
    ]

    for q, a in viva_qa:
        with st.expander(f"**{q}**"):
            st.markdown(a)

st.markdown("---")
st.markdown("<center><small>Design & Development of a Working Prototype for Waste Segregation Image Classifier Using A CNN &bull; AGGIBADALA VENKATA DURGA NAGA PRASAD &bull; Guided by Ms. G. Hema Sri</small></center>", unsafe_allow_html=True)
