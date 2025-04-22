**1. Task Objectives**
The primary goal of this project is to build a Customer Churn Prediction model using the XGBoost algorithm. The model will predict whether a customer of a bank will churn (exit) or remain a customer based on their account details, demographic information, and banking behavior.

**Key objectives:**
Preprocessing: Clean and prepare the data for modeling (handle missing values, encode categorical data, scale features).
Feature Engineering: Create additional features that help the model make better predictions (e.g., BalancePerProduct, AgeTenure).
Class Imbalance: Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset and avoid bias toward the majority class.
Model Training: Train an XGBoost classifier, which is well-known for its performance in classification tasks.
Evaluation: Assess the model's performance using metrics like accuracy, confusion matrix, and classification report.
Visualization: Visualize feature importance to understand which features contribute the most to the model's predictions.

**2. Steps to Run the Project**
**Pre-requisites:**
Python 3.x (preferably Python 3.8+)
Google Colab, Jupyter Notebook, or your local environment with the required libraries installed.

**Libraries:**
Make sure to install the following libraries if you are running the code on your local machine. You can install them using pip:
pip install xgboost imbalanced-learn scikit-learn pandas matplotlib seaborn

**Steps:**
Clone or Download the Repository: If you're using Git, clone the repository:
git clone https://github.com/GaganaShreeS/customer_churn_prediction.git

Download the Dataset: The dataset used in this project is available on Kaggle. You can download it directly from here.
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
After downloading, upload the Churn_Modelling.csv file to your working directory.

Run the Code: Open the notebook in Google Colab or your local Jupyter Notebook environment and run the cells sequentially. Ensure that the dataset (Churn_Modelling.csv) is uploaded to the environment.

Evaluation: Once the model is trained, you will get the accuracy, confusion matrix, and classification report. Feature Importance will also be plotted to help you understand which features contribute most to the model's decision-making process.

Visualization: The code will plot the feature importance of the model to visualize which features had the most influence on predicting churn.

**3. Project Structure**
Here is the structure of the project:

customer_churn_prediction/
│
├── Churn_Modelling.csv        # Dataset used for customer churn prediction
├── churn_prediction.ipynb     # Jupyter Notebook with the model code
├── requirements.txt           # List of required Python libraries
└── README.md                  # This file

**4. Code Quality and Structure**
The code is well-structured with clear and organized sections:
Data Preprocessing: Data cleaning and preparation (removing irrelevant columns, encoding categorical variables).
Feature Engineering: Creating new features that might improve model performance.
Model Training: Using XGBoost classifier for training.
Model Evaluation: Using accuracy, confusion matrix, and classification report to evaluate the model.
Feature Importance: Visualizing which features are most important.

The code is well-commented:
Each major step in the process is explained with comments for better understanding.
The function of each block of code is clarified to ensure readability and ease of modification.
The code is written to be modular and scalable, making it easy to modify and extend for other similar projects or use cases.

**5. Clean and Well-Commented Code**
The code is available in the separate file churn_prediction.ipynb in this repository.

**6. Expected Output**
Accuracy: The model's accuracy score (should aim for high accuracy, ideally above 90%).
Confusion Matrix: A matrix to visualize the performance of the model (True Positives, False Positives, True Negatives, and False Negatives).
Classification Report: A detailed report with precision, recall, and F1-score for both classes (Churned and Not Churned).
Feature Importance Plot: A bar plot to visualize the most important features used by the model.

**Contact Information**
If you have any questions or need further assistance, feel free to reach out:
Email: gaganasomu66@gmail.com
GitHub: https://github.com/GaganaShreeS






