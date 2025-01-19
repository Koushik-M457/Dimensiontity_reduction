# Dimensiontity_reduction
This project demonstrates dimensionality reduction techniques and their application in improving the performance of machine learning models. The dataset used is "Human Activity Recognition Using Smartphones," obtained from the UCI Machine Learning Repository

Project Overview

This project demonstrates the application of dimensionality reduction techniques to improve the performance of machine learning models. Using the "Human Activity Recognition Using Smartphones" dataset from the UCI Machine Learning Repository, the workflow highlights the impact of redundant and irrelevant features on model accuracy and computational efficiency. The project utilizes K-Means clustering to reduce dimensionality and evaluates the performance of a Gaussian Naive Bayes model before and after reduction.

Features

Exploratory Data Analysis (EDA): Understands the dataset structure and identifies redundant features.

Feature Encoding and Scaling: Prepares the data using Label Encoding and StandardScaler for better model compatibility.

Baseline Model Training: Trains a Gaussian Naive Bayes model on the full feature set to establish baseline metrics.

Dimensionality Reduction: Applies K-Means clustering to select representative features.

Performance Comparison: Evaluates model performance on full vs. reduced feature sets in terms of accuracy and training time.

Project Workflow

Data Loading

The dataset is downloaded from the UCI repository and preprocessed to extract feature and label data.

Exploratory Data Analysis (EDA)

Basic statistics, missing value analysis, and feature characteristics are examined.

Feature Preprocessing

Class labels are encoded using LabelEncoder.

Features are scaled using StandardScaler.

Baseline Model Training

A Gaussian Naive Bayes model is trained on the full dataset to establish baseline accuracy and training time.

Dimensionality Reduction with K-Means

K-Means clustering identifies and selects representative features.

The reduced feature set is used to train a second Gaussian Naive Bayes model.

Performance Evaluation

Accuracy and training time of the models trained on the full and reduced feature sets are compared.

Key Results

Baseline Model: The model trained on the full dataset provides baseline accuracy and training time.

Reduced Features Model: A model trained on the reduced feature set is evaluated for accuracy and computational efficiency improvements.

Prerequisites

Python 3.7+

Required Libraries:

pandas
numpy
scikit-learn
requests
beautifulsoup4

How to Run the Project

Clone the repository or download the project files.

Install the required Python libraries using:

pip install -r requirements.txt

Execute the Jupyter Notebook (Dimensionality_Reduction_.ipynb) to:

Perform EDA.

Train models with full and reduced feature sets.

Compare results.

Dataset

Name: Human Activity Recognition Using Smartphones

Source: UCI Machine Learning Repository

Project Files

Dimensionality_Reduction_.ipynb: Jupyter Notebook with the project code.

requirements.txt: List of required Python libraries.

Future Scope

Explore other dimensionality reduction techniques like PCA or t-SNE.

Apply the methodology to larger and more complex datasets.

Experiment with other machine learning models and pipelines.

Acknowledgments

UCI Machine Learning Repository for providing the dataset.

Scikit-learn library for machine learning tools and techniques.

