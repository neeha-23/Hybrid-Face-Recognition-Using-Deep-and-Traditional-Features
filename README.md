# Hybrid-Face-Recognition-Using-Deep-and-Traditional-Features
## Overview
This project presents a hybrid face recognition pipeline that integrates deep learning features (FaceNet embeddings) and traditional handcrafted features (Local Binary Patterns and Histogram of Oriented Gradients) to enhance recognition performance under real-world, unconstrained conditions.

The hybrid feature representation is evaluated using three classifiers:

Support Vector Machine (SVM)

Random Forest (RF)

XGBoost

Visualization techniques such as t-SNE clustering, ROC and PR curves, and autoencoder-based heatmaps are used for detailed model interpretation.

## Project Dependencies
This project requires a combination of deep learning, machine learning, and image processing libraries.
All essential packages — including PyTorch for model development, scikit-learn and XGBoost for classification, scikit-image and OpenCV for feature extraction, and matplotlib/seaborn for visualization — are listed in the requirements.txt file.
Please install these dependencies before running the notebook to ensure smooth execution and reproducibility.
### When setting up your environment, after putting this file in your project folder, run:

bash
Copy
Edit
pip install -r requirements.txt

## Project Structure
notebooks/HybridFaceRecognition_IA.ipynb – Main implementation notebook

models/ – Contains serialized .pkl model files (e.g., SVM, Random Forest, XGBoost models)

figures/ – Contains all important images (pipeline diagram, t-SNE plots, reconstruction heatmaps, feature visualizations)

src/ – Helper scripts for feature extraction and evaluation (optional if modularized)

README.md – Project overview

## Dataset
This project uses the Labeled Faces in the Wild (LFW) dataset.

### Note:
Due to size constraints, the dataset is not included in this repository.
Please download the LFW dataset manually from Kaggle.

Once downloaded, place the dataset in the following structure:
/content/lfw_dataset/lfw_funneled/
    person1/
        img1.jpg
        img2.jpg
    person2/
        img1.jpg
        ...
## Setup Instructions
Clone the repository.

Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
(Major libraries: facenet-pytorch, torchvision, scikit-image, scikit-learn, xgboost, matplotlib, seaborn) 3. Download the LFW dataset and organize it as shown above. 4. Open and run the Jupyter notebook HybridFaceRecognition_IA.ipynb on Google Colab or local GPU-enabled environment.

## Results Summary
Best Classifier: XGBoost

Test Accuracy: 90%

Weighted F1-Score: 0.88

t-SNE visualization showed strong cluster separation.

ROC-AUC Score: 0.94.

## Future Work
Fairness-aware learning to handle demographic bias.

Adversarial defense techniques against spoofing/deepfakes.

Privacy-preserving face recognition using federated learning.
