CREDIT CARD FRAUD DETECTION SYSTEM
==================================

OVERVIEW
--------
This project is a Machine Learning application designed to detect fraudulent credit card transactions. Due to the highly imbalanced nature of credit card fraud data, this system utilizes Logistic Regression to classify transactions as either legitimate or fraudulent. 

PROBLEM STATEMENT
-----------------
Credit card fraud is a significant problem in the financial sector. The challenge in fraud detection is the extreme class imbalance—fraudulent transactions typically represent a tiny fraction of total transactions. This project demonstrates how to preprocess imbalanced data, handle missing values, and build a reliable predictive model using Logistic Regression.

TECHNOLOGIES USED
-----------------
* Python 3.x
* Machine Learning: Scikit-Learn (Logistic Regression, Train-Test Split, Metrics)
* Data Manipulation: Pandas, NumPy
* Data Visualization: Matplotlib, Seaborn

DATASET
-------
The model is designed to be trained on a dataset containing credit card transactions. 
* Features: The dataset typically contains numerical input variables which are the result of a PCA transformation to protect user confidentiality.
* Target: Class (0 for Legitimate transaction, 1 for Fraudulent transaction).

INSTALLATION & SETUP
--------------------
1. Clone the repository:
   git clone https://github.com/your-username/credit-card-fraud-prediction.git
   cd credit-card-fraud-prediction

2. Create a virtual environment (Recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`

3. Install dependencies:
   pip install -r requirements.txt

USAGE
-----
1. Data Preprocessing & Training:
   Run the training script to load the data, handle the class imbalance (via techniques like Under-sampling or SMOTE), and train the Logistic Regression model.
   python src/train_model.py

2. Making Predictions:
   Use the prediction script to evaluate new transactions.
   python src/predict.py --input new_transactions.csv

EVALUATION METRICS
------------------
Because the dataset is highly imbalanced, standard accuracy is not a reliable metric. The model is evaluated based on:
* Precision: To measure how many of the predicted frauds are actually frauds (minimizing false positives).
* Recall: To measure how many actual frauds were successfully identified (minimizing false negatives).
* F1-Score: The harmonic mean of precision and recall.
* Confusion Matrix: To visually interpret the True Positives, True Negatives, False Positives, and False Negatives.
Author Adharsh Kumar
