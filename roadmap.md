# (a sketch of) Project Roadmap: Drone-Based Terrain Segmentation

This project plan is divided into some kind of Epics, Features, and Tasks way of work.

---

## Stage 1: Setup & Data Exploration (EDA)

*Goal: Understand the data and prepare the environment.*

### Epic: Project Configuration
- [x] **Feature:** Initialize the working environment
  - [x] **Task:** Create the folder structure.
  - [x] **Task:** Fill the readme.md and roadmap.md files.
  - [x] **Task:** Create the `requirements.txt` file.
  - [x] **Task:** Download and unpack the data from Kaggle into `data/raw/`.

### Epic: Exploratory Data Analysis (EDA)
- [x] **Feature:** Analyze the dataset
  - [x] **Task:** Create the `01_data_exploration.ipynb` notebook.
  - [x] **Task:** Class analysis: identify classes (e.g., "obstacles", "water"), count them, check pixel distribution per class (is the dataset balanced?).
  - [x] **Task:** Visualize sample pairs (image + mask) to understand the annotations.

---

## Stage 2: Preprocessing and Feature Extraction

*Goal: Transform raw images into a set of features that a machine learning model can understand.*

### Epic: Data Preprocessing
- [x] **Feature:** Create a data loading pipeline
  - [x] **Task:** Implement functions in `source/data_loader.py` to load images and masks.
  - [x] **Task:** Create a script to split the data into `train`, `val`, `test` sets and save them in `data/processed/`.

### Epic: Feature Engineering
- [ ] **Feature:** Implement feature extractors in `source/feature_extractor.py`
  - [ ] **Task:** Extract color-based features (e.g., values from HSV or L*a*b* color spaces).
  - [ ] **Task:** Extract texture-based features (e.g., Gabor filters, Local Binary Patterns (LBP), or Gray-Level Co-occurrence Matrix (GLCM)).
  - [ ] **Task:** Extract edge-based features (e.g., response from Sobel or Canny filters).
- [ ] **Feature:** Create the feature vector
  - [ ] **Task:** Create a function that generates a single, flat feature vector for each pixel in the image (e.g., `[R, G, B, H, S, V, Gabor1, Gabor2, ..., LBP, SobelX, SobelY]`).

---

## Stage 3: Building and Training Segmentation Models

*Goal: Train classic ML models to classify each pixel based on its extracted features.*

### Epic: Prepare Training Data
- [ ] **Feature:** Transform data for Scikit-learn
  - [ ] **Task:** Create the training matrix `X` (number_of_pixels_in_train_set, number_of_features).
  - [ ] **Task:** Create the label vector `y` (number_of_pixels_in_train_set, 1).
  - [ ] **Task:** *Optional:* Investigate dimensionality reduction (PCA) on the feature vector.
  - [ ] **Task:** *Optional:* Investigate segmentation at the "superpixel" level (e.g., SLIC algorithm) instead of individual pixels to reduce computational cost.

### Epic: Model Training
- [ ] **Feature:** Implement and train models in `source/models.py` and `03_model_training_dev.ipynb`
  - [ ] **Task:** Implement and train a K-Nearest Neighbors (KNN) model.
  - [ ] **Task:** Implement and train a Support Vector Machine (SVM) model.
  - [ ] **Task:** Implement and train a Random Forest model.
  - [ ] **Task:** Save (serialize) the best models to files (e.g., `.pkl` or `.joblib`).

---

## Stage 4: Evaluation and Results Analysis

*Goal: Objectively assess the segmentation quality and visually inspect the results.*

### Epic: Evaluation Metrics
- [ ] **Feature:** Implement metrics in `source/evaluate.py`
  - [ ] **Task:** Implement the Pixel Accuracy metric.
  - [ ] **Task:** Implement the Intersection over Union (IoU) metric for each class and the mean (mIoU).
  - [ ] **Task:** Implement Confusion Matrix generation.

### Epic: Results Analysis
- [ ] **Feature:** Generate and analyze predictions
  - [ ] **Task:** Load the saved models and generate prediction masks for the `test` set.
  - [ ] **Task:** Calculate all metrics for each of the tested models.
  - [ ] **Task:** Create comparative charts (e.g., bar plots) for metrics (mIoU, Accuracy) in `04_results_analysis.ipynb`.
  - [ ] **Task:** Create visualizations in `source/visualization.py` showing side-by-side: [Original Image] | [Ground Truth Mask] | [Predicted Mask]. Save them in `results/plots/`.

---

## Stage 5: Finalization and Reporting

*Goal: Prepare the final report and clean code.*

### Epic: Documentation
- [ ] **Feature:** Update `readme.md`
  - [ ] **Task:** Add a brief description of the best-performing model.
  - [ ] **Task:** Insert a table with the results (mIoU, Accuracy) for all models.
  - [ ] **Task:** Add a few of the best prediction visualizations.
- [ ] **Feature:** Code cleanup
  - [ ] **Task:** Add comments and docstrings to functions in `source/`.
  - [ ] **Task:** Ensure notebooks are clean and runnable from start to finish.

### Epic: Final Report
- [ ] **Feature:** Create the report (e.g., in `results/report/report.pdf`)
  - [ ] **Task:** Describe the problem, dataset, and the experiments conducted.
  - [ ] **Task:** Provide a detailed discussion of the feature extraction methods used.
  - [ ] **Task:** Present and analyze the obtained results (metrics, confusion matrices, visualizations).
  - [ ] **Task:** Formulate conclusions (which features were most important? which model performed best? what are the limitations and potential directions for future work?).
