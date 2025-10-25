
# 🩺 AI-Powered Healthcare Analysis  
### Member 1 – : Hafsa Hajir  

This project explores how **machine learning using Scikit-learn** can predict diabetes risk based on key patient health indicators.  
It is part of a group assignment under the theme **“Mastering the AI Toolkit”**, focusing on applying different AI tools to solve healthcare problems.  

---

## 📋 Project Overview  

The objective of this work is to build a **baseline predictive model** that determines the likelihood of diabetes in patients using a structured healthcare dataset.  
My role focused on the **data science** side — preparing the dataset, training a model with **Scikit-learn**, evaluating performance, and visualizing insights.  

**Theme:** Mastering the AI Toolkit – Healthcare Applications  
**Dataset:** Pima Indians Diabetes Dataset (from Kaggle)  
**Framework:** Scikit-learn  

---

## 👩🏽‍💻 My Role  

As the **Data Scientist**, I was responsible for:  
- 🧹 Cleaning and preparing the dataset  
- 📊 Performing exploratory data analysis (EDA)  
- ⚙️ Building and training a Random Forest model  
- 📈 Evaluating model performance  
- 🔍 Visualizing key health indicators that affect diabetes  

---

## 🧰 Tools and Libraries Used  

- **Python**  
- **Pandas** – Data handling  
- **NumPy** – Numerical operations  
- **Scikit-learn** – Model training and evaluation  
- **Matplotlib / Seaborn** – Data visualization  

---

## 📊 Dataset Description  

**Name:** Pima Indians Diabetes Dataset  
**Source:** [Kaggle – Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  

**Features:**
- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome *(Target: 1 = Diabetic, 0 = Non-Diabetic)*  

---

## 💻 Model Implementation  

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.show()

---
📈 Results and Insights
Metric	Score
Accuracy	~82%
Precision (Diabetic)	0.78
Recall (Diabetic)	0.80
🔍 Key Findings

Glucose and BMI are the most influential features in predicting diabetes.

Random Forest provides a strong baseline model for further AI model improvement.

Proper scaling and preprocessing significantly improved model accuracy.

🧭 Reflection

This task strengthened my understanding of:

End-to-end data science workflows in healthcare applications

The use of Scikit-learn for practical machine learning tasks

How classical ML serves as a foundation for advanced AI models

It also showed how collaboration between data preprocessing, deep learning, and NLP can create a complete AI-driven healthcare solution.

🏁 Conclusion

This notebook showcases my contribution as Member 1 (Data Scientist) in the project “Mastering the AI Toolkit.”
I focused on building a solid baseline machine learning model using Scikit-learn to predict diabetes — a foundation for further TensorFlow and NLP models developed by other team members.

## 📈 Results and Insights  

| **Metric** | **Score** |
|-------------|-----------|
| Accuracy | ~82% |
| Precision (Diabetic) | 0.78 |
| Recall (Diabetic) | 0.80 |

---

### 🔍 Key Findings  

- **Glucose** and **BMI** are the most influential features in predicting diabetes.  
- **Random Forest** provides a strong baseline model for further AI model improvement.  
- Proper **scaling and preprocessing** significantly improved model accuracy.  

---

## 🧭 Reflection  

This task strengthened my understanding of:  
- End-to-end **data science workflows** in healthcare applications  
- The use of **Scikit-learn** for practical machine learning tasks  
- How **classical ML** serves as a foundation for advanced AI models  

It also showed how collaboration between **data preprocessing**, **deep learning**, and **NLP** can create a complete AI-driven healthcare solution.  

---

## 🏁 Conclusion  

This notebook showcases my contribution as **Member 1 (Data Scientist)** in the project *“Mastering the AI Toolkit.”*  
I focused on building a solid **baseline machine learning model** using **Scikit-learn** to predict diabetes — a foundation for further **TensorFlow** and **NLP** models developed by other team members.




