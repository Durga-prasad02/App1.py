# ACADEMIC PROJECT SYNOPSIS

## ♻️ Design and Development of a Working Prototype for Waste Segregation Image Classifier Using a CNN Trained on Waste-Category Images

---

### 1. General Project Information
- **Project Title:** Design and Development of a Working Prototype for Waste Segregation Image Classifier Using a CNN Trained on Waste-Category Images
- **Candidate / Student Name:** AGGIBADALA VENKATA DURGA NAGA PRASAD
- **Supervisor / Project Guide:** Ms. G. Hema Sri
- **Domain:** Deep Learning & Computer Vision
- **Department:** Computer Science & Engineering / Information Technology
- **Keywords:** waste, segregation, image, classifier, cnn, trained, smart-bin, computer vision, deep learning, circular economy

---

### 2. Abstract
This work presents a Convolutional Neural Network (CNN) trained on waste-category images to sort waste images into recyclable and compostable categories, addressing the problem of municipal waste segregation through an automated, data-driven computer vision approach. Rapid urbanization and unsustainable consumption habits have precipitated a global solid waste management crisis. Manual segregation at waste disposal points is labor-intensive, hazardous to human sanitary workers, and plagued by severe cross-contamination. 

To overcome these challenges, the developed system trains a specialized multi-stage CNN architecture alongside transfer learning benchmarks on municipal waste image repositories. Domain-tailored image pre-processing and data augmentation steps—such as rotational invariance, illumination perturbation, and affine transformations—are applied to simulate unpredictable smart-bin internal lighting and crumpled item geometries. The final deliverable is an end-to-end **Smart-Bin Sorting Assistant** working prototype featuring real-time visual classification, servo motor flap actuation simulation, contamination advisories, and municipal telemetry logging, aimed at making the solution usable by real municipal and commercial stakeholders.

---

### 3. Problem Statement
Municipal Solid Waste (MSW) management represents one of the most critical environmental imperatives globally. Conventional waste management suffers from several severe limitations:
1. **Low Segregation Compliance at Source:** Despite municipal guidelines, residential and public waste bins frequently suffer from mixed disposal (e.g., placing half-filled soda cans into paper bins), causing cross-contamination that renders entire batches non-recyclable.
2. **Occupational Hazards in Manual Sorting:** Manual sorting at transfer stations exposes sanitation workers to chemical toxicants, airborne pathogens, sharp glass, and biohazardous waste.
3. **High Intra-Class Variance in Waste Items:** An item of plastic waste may present as a clear cylindrical bottle, a crumpled translucent shopping bag, or an opaque colorful shampoo flask. Traditional rule-based or feature-engineered machine learning algorithms (e.g., SIFT, HOG + SVM) struggle to generalize across diverse visual deformations.

---

### 4. Objectives of the Work
- **Design and Train a Custom Deep CNN (`WasteCNN`):** Construct a multi-stage convolutional architecture with Batch Normalization and Dropout regularization optimized for fine-grained multi-class waste categorization.
- **Implement Domain-Tailored Preprocessing & Augmentation:** Apply color jittering, random affine shifts, and multi-scale normalization to mimic optical conditions inside municipal bins.
- **Categorize Multi-Class Municipal Waste:** Accurately classify items across standard municipal categories: *Cardboard, Glass, Metal, Paper, Plastic, and Organic/Food Waste*.
- **Develop a Working Prototype (Smart-Bin Sorting Assistant):** Engineer an interactive web application that integrates optical image ingestion (webcam and image uploads), predicts class probabilities, visualizes physical servo diverter flap routing, calculates environmental carbon offsets, and provides actionable contamination tips.
- **Municipal Telemetry & Audit Reporting:** Provide audit logging of sorted items, fill-level telemetry simulation, and exportable CSV reports for municipal sanitation authorities.

---

### 5. System Architecture & Methodology
```
 +-------------------------------------------------------------------------------+
 |                              SYSTEM ARCHITECTURE                              |
 +-------------------------------------------------------------------------------+
 |                                                                               |
 |  [ Optical Ingestion ]                                                        |
 |         │  (Webcam / Gallery / Image Upload)                                  |
 |         ▼                                                                     |
 |  [ Preprocessing & Normalization ]                                            |
 |         │  Resize (224×224), ImageNet Standardization, Domain Augmentation   |
 |         ▼                                                                     |
 |  [ Deep CNN Classifier (WasteCNN) ]                                           |
 |         │  Conv2D + BatchNorm + ReLU + MaxPool (4 Stages)                     |
 |         │  Dense Representation & Softmax Classification Head                 |
 |         ▼                                                                     |
 |  [ Sorting Logic & Decision Engine ]                                          |
 |         │  Class Probabilities & Confidence Verification                      |
 |         ▼                                                                     |
 |  [ Smart-Bin Assistant Working Prototype ]                                    |
 |         ├── Physical Bin Chute Routing (Blue/Yellow/Gray/Orange/Green)        |
 |         ├── Servo Motor Flap Actuation (30° - 180° Rotation)                  |
 |         ├── Contamination Advisory & Cleanliness Tip                          |
 |         ├── Environmental Impact Calculation (CO₂ & Energy Offsets)           |
 |         └── Municipal Telemetry Dashboard & CSV Audit Trail                   |
 +-------------------------------------------------------------------------------+
```

---

### 6. Hardware & Software Requirements

#### Software Specifications
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Programming Language:** Python 3.9+
- **Deep Learning Framework:** PyTorch / TorchVision (or TensorFlow / Keras)
- **Computer Vision & Image Processing:** Pillow (PIL), NumPy, OpenCV
- **Web Application / Prototype Framework:** Streamlit
- **Data Analytics & Visualization:** Pandas, Matplotlib, Seaborn, Altair

#### Hardware Specifications (Recommended)
- **Development Workstation:** Intel Core i5/i7 or AMD Ryzen 5/7, 8GB+ RAM, NVIDIA GPU (optional for accelerated training).
- **Embedded Smart-Bin Target Hardware (Edge Deployment):** Raspberry Pi 4 Model B (4GB RAM) or NVIDIA Jetson Nano / Orin Nano, wide-angle 5MP camera module, MG996R high-torque servo motor, HC-SR04 ultrasonic sensors.

---

### 7. Expected Deliverables & Outcomes
1. **Fully Functional Working Prototype:** Interactive Streamlit dashboard enabling real-time waste classification, visual bin routing, and IoT actuator simulation.
2. **Modular Codebase:** Standalone modules for model definition, dataset loading, preprocessing, metric evaluation, and training execution.
3. **High Empirical Performance:** Benchmark classification accuracy $\ge 95.0\%$, macro F1-score $\ge 0.95$, and inference latency $< 25 \text{ ms}$ on edge-ready architectures.
4. **Comprehensive Academic Documentation:** Complete project report, literature review, viva voce preparation manual, and dataset usage guidelines.

---

**Candidate:** AGGIBADALA VENKATA DURGA NAGA PRASAD  
**Supervisor / Guide:** Ms. G. Hema Sri  
**Academic Year:** 2025–2026
