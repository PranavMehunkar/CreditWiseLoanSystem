<h1>CreditWise Loan System</h1>
CreditWise Loan System is a machine learning project that predicts loan approval outcomes based on applicant financial, personal, and credit-related information. The goal is to analyze risk factors and build a reliable prediction system using supervised learning techniques.

<h2>Project Structure</h2>
train_model.ipynb – Model for training and evaluation
LoanLens_Full.ipynb – Complete end-to-end ML pipeline
loan_approval_data.csv – Dataset used for training and evaluation

Models Used
The following models were trained and evaluated:

Logistic Regression
K-Nearest Neighbors (KNN)
Naive Bayes
Model performance was compared using accuracy, precision, recall, F1-score, and confusion matrix.

Model Comparison
Model	Accuracy (%)	Precision (%)	Recall (%)	F1-Score (%)
Logistic Regression	87.0	78.69	78.69	78.69
K-Nearest Neighbors	75.0	62.22	45.90	52.83
Naive Bayes	86.5	80.36	73.77	76.92
“Although Logistic Regression performed best overall, Naive Bayes achieved higher precision, making it useful in scenarios where minimizing false loan approvals is critical.”

Final Model Selection
Based on comparative evaluation across multiple metrics, Naive Bayes demonstrated the most consistent performance and was selected as the final model for deployment.

Technologies Used
Python
Pandas, NumPy
Scikit-learn
Jupyter Notebook
Future Work
Deploy the final model using Streamlit
Improve feature engineering
Add model interpretability and UI enhancements
Key Learnings
During this project, I learned and applied several important machine learning and data preprocessing concepts, including:

🚀 Live Demo
LoanLens is a Streamlit-based loan approval prediction application powered by machine learning.
Try the live app here:

🔗 https://loanlens-loanpredictor.streamlit.app/

Handling missing values using SimpleImputer with appropriate strategies for numerical and categorical features
Encoding categorical variables using LabelEncoder and OneHotEncoder based on feature type
Performing correlation analysis and visualizing relationships between numerical features using correlation heatmaps
Understanding the impact of feature scaling using StandardScaler
Comparing multiple machine learning models to select the most suitable one for deployment
