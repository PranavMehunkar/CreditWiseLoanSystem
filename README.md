<h1>CreditWise Loan System</h1>
CreditWise Loan System is a machine learning project that predicts loan approval outcomes based on applicant financial, personal, and credit-related information. The goal is to analyze risk factors and build a reliable prediction system using supervised learning techniques.

<h2>Project Structure</h2>
<ul><li>train_model.ipynb – Model for training and evaluation</li></ul>
<ul><li>app.py – Complete end-to-end ML pipeline</li></ul>
<ul><li>loan_approval_data.csv – Dataset used for training and evaluation</li></ul>

<h2>Models Used</h2>
The following models were trained and evaluated:<br>
<ul><li>Logistic Regression</li></ul>
<ul><li>K-Nearest Neighbors (KNN)</li></ul>
<ul><li>Naive Bayes</li></ul>
Model performance was compared using accuracy, precision, recall, F1-score, and confusion matrix.

<h2>Model Comparison</h2>
<table>
<tr><th>Model</th><th>Accuracy (%)</th>	<th>Precision (%)</th><th>Recall (%)</th>	<th>F1-Score (%)</th></tr>
<tr><td>Logistic Regression</td>	<td><b>82.0</b></td>	<td>77.74</td>	<td><b>82.0</b></td>	<td><b>79.75</b></td></tr>
<tr><td>K-Nearest Neighbors</td>	<td>71.5</td>	<td>67.21</td>	<td>71.5</td>	<td>69.04</td></tr>
<tr><td>Naive Bayes</td>	<td>79.0</td>	<td><b>78.23</b></td>	<td>79.0</td>	<td>78.12</td></tr>
</table>
<b>“Although Logistic Regression performed best overall, Naive Bayes achieved higher precision, making it useful in scenarios where minimizing false loan approvals is critical.”</b>

<h2>Final Model Selection</h2>
Based on comparative evaluation across multiple metrics, <b>Naive Bayes</b> demonstrated the most consistent performance and was selected as the final model for deployment.

<h2>Technologies Used</h2>
<ul><li>Python</li></ul>
<ul><li>Pandas, NumPy</li></ul>
<ul><li>Scikit-learn</li></ul>
<ul><li>Jupyter Lab</li></ul>

<h2>🚀Live Demo</h2>
CreditWise Loan System is a Streamlit-based loan approval prediction application powered by machine learning.<br>
Try the live app here:

🔗 https://creditwiseloansystem-4kaybza5i4rcy7e9xf49dh.streamlit.app/

<h2>Future Work</h2>
<ul><li>Deploy the final model using Streamlit</li></ul>
<ul><li>Improve feature engineering</li></ul>
<ul><li>Add model interpretability and UI enhancements</li></ul>

<h2>Key Learnings</h2>
During this project, I learned and applied several important machine learning and data preprocessing concepts, including:
<ul><li>Handling missing values using <b>SimpleImputer</b> with appropriate strategies for numerical and categorical features</li></ul>
<ul><li>Encoding categorical variables using <b>LabelEncoder</b> and <b>OneHotEncoder</b> based on feature type</li></ul>
<ul><li>Performing <b>correlation analysis</b> and visualizing relationships between numerical features using correlation heatmaps</li></ul>
<ul><li>Understanding the impact of feature scaling using <b>StandardScaler</b></li></ul>
<ul><li>Comparing multiple machine learning models to select the most suitable one for deployment</li></ul>
