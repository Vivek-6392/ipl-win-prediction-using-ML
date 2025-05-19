# 🏏 IPL Win Prediction Using Machine Learning

## 📘 Overview

This project uses **machine learning algorithms** to predict the outcome of Indian Premier League (IPL) matches based on historical match data. By analyzing features such as team statistics, toss results, venue, and match history, we build a predictive model that estimates which team is more likely to win a given match.

---

## 🎯 Objectives

* Predict the winner of IPL matches using past data.
* Analyze key factors that influence match outcomes (e.g., toss, venue, team strength).
* Explore and compare the performance of various machine learning models.

---

## 📂 Dataset

Dataset contains historical IPL match data with the following key features:

* `Season`
* `Date`
* `Team1`, `Team2`
* `Toss Winner`, `Toss Decision`
* `Venue`
* `Winner`
* `City`, `Umpires`
* (Optional) Player performance, net run rate, etc.

📌 **Data Source**:

* [Kaggle IPL Dataset](https://www.kaggle.com/datasets/rahulbansal/ipl)
* Official IPL statistics

---

## 🛠️ Tools & Technologies

* **Python 3.x**
* **Pandas**, **NumPy** – Data handling
* **Matplotlib**, **Seaborn** – Data visualization
* **Scikit-learn** – ML models and evaluation
* **Jupyter Notebook / VS Code**

---

## 🤖 Machine Learning Models Used

* **Logistic Regression**
* **Random Forest Classifier**
* **Decision Tree Classifier**
* **Gradient Boosting Classifier**
* **Support Vector Machine**

---

## ⚙️ Project Workflow

1. **Data Preprocessing**

   * Handle missing values
   * Convert categorical data (Label Encoding or One-Hot Encoding)
   * Feature selection (e.g., remove `umpires`, `ID`, etc.)

2. **Exploratory Data Analysis (EDA)**

   * Most successful teams
   * Impact of toss on result
   * Venue-wise win percentage

3. **Model Training**

   * Train/test split
   * Fit models using training data
   * Predict match outcomes

4. **Evaluation**

   * Accuracy
   * Confusion Matrix
   * Precision, Recall, F1-Score

5. **Prediction Interface** *(Optional)*
   A simple input form to predict match outcome given team names, venue, and toss result.

---

## 📊 Visualizations

* Toss winner vs match winner correlation
* Venue-wise win distribution
* Team performance trends over seasons
* Feature importance plots

---

## 💡 Future Enhancements

* Use live IPL data via API for real-time predictions.
* Add player-level performance data for improved accuracy.
* Build a web app using Flask or Streamlit.
* Use ensemble models or XGBoost for better performance.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss your ideas or improvements.

---

## 🙏 Acknowledgements

* Kaggle for the IPL datasets
* IPL T20 official stats
* Scikit-learn and the Python ML community

---
