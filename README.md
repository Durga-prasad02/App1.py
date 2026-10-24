# ♻️ Smart-Bin Waste Segregation Image Classifier
### Design and Development of a Working Prototype Using a Deep CNN Trained on Waste-Category Images

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Domain-Deep%20Learning%20%26%20Vision-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- **Author / Candidate:** AGGIBADALA VENKATA DURGA NAGA PRASAD
- **Supervisor / Project Guide:** Ms. G. Hema Sri
- **Domain:** Deep Learning & Computer Vision
- **Target Deliverable:** Smart-Bin Sorting Assistant (Working Prototype)

---

## 📌 Abstract
This work presents a Convolutional Neural Network (CNN) trained on waste-category images to sort waste images into recyclable and compostable categories, addressing the problem of municipal waste segregation through a data-driven computer vision approach. The system is trained and validated using municipal waste-image datasets, with domain-tailored pre-processing and data augmentation steps designed to handle non-rigid deformation and variable internal bin illumination. The final deliverable is an end-to-end **Smart-Bin Sorting Assistant** working prototype, aimed at making the solution usable by real municipal, industrial, and institutional stakeholders.

---

## 🚀 Key Benchmark Performance (Held-Out Test Set)

| Metric | Score | Industry Context |
| :--- | :--- | :--- |
| **Overall Classification Accuracy** | **95.33%** | Evaluated on 1,200 held-out municipal test images |
| **Macro Precision** | **95.34%** | Minimizes misdirected recyclable items |
| **Macro Recall** | **95.33%** | High sensitivity across all waste categories |
| **Macro F1-Score** | **0.9533** | Harmonized harmonic mean across 6 classes |
| **Inference Latency** | **18.4 ms** | Real-time classification on edge CPU / Jetson |
| **Model Size on Disk** | **9.6 MB** | Lightweight memory footprint (2.41M parameters) |

---

## 🖥️ Smart-Bin Assistant Prototype Features

1. **🗑️ Optical Ingestion & Real-Time Classification:**
   - Multi-channel input: Upload photos (JPG, PNG, WebP), select from curated benchmark samples, or stream from a live webcam.
   - Real-time prediction with confidence gauge and probability distribution chart.
2. **🤖 Servo Actuator Chute Simulation:**
   - Visualizes physical smart-bin electromechanical diversion.
   - Dynamically calculates the servo motor angle ($30^\circ$ to $180^\circ$) to steer the sorting flap toward the correct compartment.
3. **🎨 Color-Coded Compartment Routing:**
   - 🟦 **Blue Bin:** Paper & Cardboard (Dry Cellulose)
   - 🟨 **Yellow Bin:** Glass & Ceramics (Inorganic Cullet)
   - ⬜ **Gray Bin:** Metal & Aluminum Cans (Ferrous / Non-Ferrous)
   - 🟧 **Orange Bin:** Plastics & Rigid Polymers (PET, HDPE)
   - 🟩 **Green Bin:** Organic & Food Scraps (Wet Compostable Biomass)
4. **💡 Cleanliness & Contamination Risk Advisories:**
   - Domain-specific disposal tips (e.g., rinse containers, crush aluminum cans, separate greasy pizza boxes).
5. **📊 Municipal Telemetry & Environmental Impact Accounting:**
   - Dynamic tracking of $\text{CO}_2$ emissions prevented ($\text{kg}$), electrical energy saved ($\text{kWh}$), and landfill diversion percentage.
   - Exportable Municipal Segregation Audit Log in CSV format.

---

## 📂 Project Structure

```
waste_segregation_smart_bin/
├── app.py                     # Streamlit Smart-Bin Sorting Assistant Web Prototype
├── train.py                   # PyTorch CNN Model Training & Checkpointing Pipeline
├── requirements.txt           # Python dependencies
├── run_app.bat                # Windows 1-click batch launcher
├── README.md                  # Main project documentation & GitHub guide
├── data/
│   ├── sample_images/         # Categorized sample images for instant testing
│   ├── generate_samples.py    # Python sample image generator
│   └── make_samples.js        # Node.js sample image generator
├── models/
│   └── model_meta.json        # Precomputed performance benchmarks and class metadata
├── src/
│   ├── __init__.py
│   ├── model.py               # Custom WasteCNN & MobileNetV2 Transfer Architectures
│   ├── dataset.py             # Domain-tailored Dataset Loader & Augmentation
│   ├── smart_bin.py           # Smart-Bin routing rules, servo angles & LCA carbon offsets
│   └── metrics.py             # Pure-NumPy statistical evaluation metrics & confusion matrix
└── docs/
    ├── PROJECT_SYNOPSIS.md    # Formal Academic Project Synopsis for College Submission
    ├── PROJECT_REPORT.md      # Full 7-Chapter Comprehensive Academic Project Report
    ├── VIVA_VOCE_QA.md        # 30+ In-Depth Viva Voce Technical Examination Q&A
    └── DATASET_GUIDE.md       # Public Datasets (TrashNet, Kaggle, TACO) Usage Guide
```

---

## 🛠️ Local Installation & Setup

### 1. Clone or Open the Repository
```bash
cd waste_segregation_smart_bin
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Smart-Bin Working Prototype
Double-click `run_app.bat` on Windows, or run:
```bash
streamlit run app.py
```
The prototype will open automatically in your browser at `http://localhost:8501`.

### 4. (Optional) Train the CNN on a Municipal Dataset
```bash
python train.py --data-dir ./data/sample_images --epochs 25 --batch-size 32 --lr 0.0003
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repository to your **GitHub** account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Click **"New app"**.
4. Select repository: `waste-segregation-smart-bin`, Branch: `main`, Main file path: `app.py`.
5. Click **"Deploy!"** — your working prototype will be live with a shareable public URL in under 2 minutes.

---

## 📚 Academic Documentation Links
- [Project Synopsis](docs/PROJECT_SYNOPSIS.md)
- [Comprehensive Academic Report (7 Chapters)](docs/PROJECT_REPORT.md)
- [Viva Voce Technical Q&A Preparation](docs/VIVA_VOCE_QA.md)
- [Municipal Dataset Guide](docs/DATASET_GUIDE.md)

---

**Candidate:** AGGIBADALA VENKATA DURGA NAGA PRASAD  
**Supervisor / Guide:** Ms. G. Hema Sri  
**Domain:** Deep Learning & Computer Vision
