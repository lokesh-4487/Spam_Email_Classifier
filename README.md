# Spam_Email_Classifier#
Spam Email Classification using Machine Learning
Overview

This project focuses on building a machine learning model that can classify text messages or emails as Spam or Not Spam (Ham). The goal of this project is to understand how Natural Language Processing (NLP) techniques work in real-world applications and how machine learning algorithms can be trained on textual data.

Spam detection is a common real-life problem used in email services, messaging platforms, and online systems. Through this project, the entire pipeline is implemented — from data preprocessing to model evaluation and prediction on custom inputs.

Project Objective

The main objective of this project is to:

Understand text preprocessing techniques

Convert text into numerical features

Train a classification model

Evaluate model performance

Predict new unseen messages

This project demonstrates how raw text data can be transformed into structured data that machine learning models can understand.

Technologies and Libraries Used

The project is implemented using Python and the following libraries:

Pandas – For handling and processing the dataset

NumPy – For numerical operations

Scikit-learn – For machine learning model training and evaluation

Matplotlib (optional) – For visualization

Seaborn (optional) – For confusion matrix visualization

The model uses TF-IDF Vectorization for feature extraction and Multinomial Naive Bayes for classification.

Dataset

The dataset used is the SMS Spam Collection Dataset. It contains labeled messages categorized as:

ham → Not Spam

spam → Spam

The dataset is first cleaned and reformatted to keep only relevant columns. Labels are converted into numerical format (0 and 1) for model training.

Workflow

The project follows a structured machine learning pipeline:

1. Data Loading

The dataset is loaded using Pandas and unnecessary columns are removed.

2. Data Preprocessing

Renaming columns for clarity

Converting categorical labels into numeric format

Splitting data into training and testing sets

3. Text Vectorization

Text data cannot be directly used in machine learning models. Therefore, TF-IDF (Term Frequency – Inverse Document Frequency) is used to convert text into numerical feature vectors.

TF-IDF helps assign importance to words based on how frequently they appear in a message relative to the entire dataset.

4. Model Training

A Multinomial Naive Bayes classifier is used because it performs well for text classification tasks and works efficiently with word frequency features.

5. Model Evaluation

The trained model is evaluated using:

Accuracy Score

Confusion Matrix

This helps measure how well the model distinguishes between spam and non-spam messages.

6. Prediction on Custom Input

The system allows the user to input their own message and instantly checks whether it is Spam or Not Spam.

Results
OUTPUT :
<img width="1366" height="730" alt="Image" src="https://github.com/user-attachments/assets/5914d5cb-bbf7-45bf-9e8d-2f97f337fdd6" />
The model achieves high accuracy (around 95–97% depending on train-test split). The confusion matrix shows strong performance in identifying both spam and legitimate messages with minimal misclassification.
OU
