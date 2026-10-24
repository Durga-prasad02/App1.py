# DATASET USAGE AND ACQUISITION GUIDE
## Municipal Solid Waste Image Classification Benchmarks

**Candidate:** AGGIBADALA VENKATA DURGA NAGA PRASAD  
**Supervisor / Guide:** Ms. G. Hema Sri  
**Domain:** Deep Learning & Computer Vision  

---

### 1. Overview of Municipal Waste Datasets
The Waste Segregation Image Classifier is engineered to train and evaluate on open-access municipal solid waste image repositories. Below are the primary benchmark datasets utilized in the literature:

#### A. TrashNet Dataset (Stanford University)
- **Authors:** Gary Thung and Mindy Yang (Stanford CS229)
- **Size:** 2,527 images
- **Classes:** 6 classes (Glass: 501, Paper: 594, Cardboard: 403, Plastic: 482, Metal: 410, Trash: 137)
- **Image Resolution:** $512 \times 384$ pixels taken on white desktop background
- **Access Link:** [GitHub: garythung/trashnet](https://github.com/garythung/trashnet)

#### B. Kaggle Waste Classification Dataset
- **Authors:** Sashaank Sekhar et al.
- **Size:** 25,077 images
- **Classes:** Binary (Organic vs. Recyclable) + Multiclass variants
- **Access Link:** [Kaggle Waste Classification](https://www.kaggle.com/datasets/techsash/waste-classification-data)

#### C. TACO (Trash Annotations in Context)
- **Authors:** Pedro F. Proença & Pedro Simões (2020)
- **Size:** 1,500+ high-resolution in-the-wild images with polygon masks
- **Classes:** 60 fine-grained municipal waste categories
- **Access Link:** [TACO Dataset Official Site](http://tacodataset.org/)

---

### 2. Recommended Directory Structure
To train the WasteCNN on any downloaded dataset, arrange the directory structure in standard `ImageFolder` format:

```
waste_segregation_smart_bin/
└── data/
    └── municipal_waste_dataset/
        ├── Cardboard/
        │   ├── card_001.jpg
        │   └── card_002.jpg
        ├── Glass/
        │   ├── glass_001.jpg
        │   └── glass_002.jpg
        ├── Metal/
        │   ├── metal_001.jpg
        │   └── metal_002.jpg
        ├── Paper/
        │   ├── paper_001.jpg
        │   └── paper_002.jpg
        ├── Plastic/
        │   ├── plastic_001.jpg
        │   └── plastic_002.jpg
        └── Organic/
            ├── organic_001.jpg
            └── organic_002.jpg
```

---

### 3. Step-by-Step Dataset Download via Kaggle CLI

If you have a Kaggle account and API token (`kaggle.json`):

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Download TrashNet Dataset:**
   ```bash
   kaggle datasets download -d asdasdasasdas/garbage-classification
   ```

3. **Extract Archive into Data Directory:**
   ```bash
   unzip garbage-classification.zip -d data/garbage_classification/
   ```

4. **Verify Class Folders:**
   Ensure folder names match the 6 target categories (`Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Organic`).

---

### 4. Running the Training Pipeline
Once your dataset is downloaded into `data/garbage_classification`, invoke the training script:

```bash
# Train Custom WasteCNN (Default 4-Stage Deep Architecture)
python train.py --data-dir ./data/garbage_classification --epochs 25 --batch-size 32 --lr 0.0003 --model wastecnn

# Or Train Transfer Learning Baseline (MobileNetV2)
python train.py --data-dir ./data/garbage_classification --epochs 20 --batch-size 32 --lr 0.0001 --model mobilenetv2
```

The script will automatically:
- Split the dataset into 80% Training and 20% Held-out Validation
- Apply domain-tailored data augmentations (flips, rotations, color jitter, affine translations)
- Log loss and validation accuracy per epoch
- Save the optimal model weights to `models/best_waste_cnn.pth`
- Export training curves to `models/training_history.json`

---

### 5. Built-in Instant Testing Mode (Synthetic Dataset)
For rapid verification without downloading gigabytes of external image archives, this repository includes an automated synthetic sample generator (`data/generate_samples.py`).

Run:
```bash
python data/generate_samples.py
```
This produces categorized benchmark sample images under `data/sample_images/` enabling immediate testing of the **Smart-Bin Sorting Assistant** prototype out-of-the-box.
