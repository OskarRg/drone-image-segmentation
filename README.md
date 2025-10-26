# Terrain Segmentation from Drone Imagery

## Project Goal

The goal of this project is to implement and evaluate classical image processing and machine learning methods for the task of semantic terrain segmentation based on aerial images acquired from a drone.

The project focuses on extracting hand-crafted features (color, texture) and using them to train classification models (e.g., Random Forest, SVM) to predict the class of each pixel.

## Dataset

This project uses the **Semantic Segmentation Drone Dataset** available on Kaggle:
[https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset)

The dataset contains images and their corresponding segmentation masks with various terrain classes.

## Project Structure

A standard data science project structure is used, separating raw data, processed data, analysis notebooks (`notebooks/`), source code (`source/`), and results.

* `data/`: Contains `raw/` and `processed/` subfolders.
* `notebooks/`: For Jupyter notebooks (EDA, prototyping) (tbd).
* `source/`: For reusable Python scripts (`.py`) like `data_loader.py`, `feature_extractor.py`, `evaluate.py`.
* `results/`: For saved plots, predicted masks, and the final report.

## Installation

1.  Clone the repository.
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset) and place its contents (images and masks) into the `data/raw/` folder.

## Usage

*(To be filled in after implementation)*

Example of how to run the full pipeline (processing, training, evaluation):

```bash
python main.py
```

Individual steps can also be tracked and run manually in the Jupyter notebooks located in the `notebooks/` folder. (not prepared yet)

## Results

*(To be filled in after completing Stage 4)*

### Model Comparison

| Model | mIoU (Mean IoU) | Pixel Accuracy | 
| ----- | ----- | ----- | 
| KNN | \- | \- | 
| SVM (LinearSVC) | \- | \- | 
| Random Forest | \- | \- | 

### Sample Predictions

*(Insert visualizations here \[Original | Ground Truth | Prediction\])*
