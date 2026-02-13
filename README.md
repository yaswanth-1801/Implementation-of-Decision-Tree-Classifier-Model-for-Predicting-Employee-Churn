# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create the employee dataset and separate the features (Age, Salary, YearsAtCompany, JobSatisfaction) and target variable (Churn).
2. Split the dataset into training and testing sets.
3. Create and train the Decision Tree Classifier using entropy as the criterion
4. Predict the churn values for the test data.
5. Evaluate the model using accuracy and classification report, and visualize the decision tree.

## Program:
```
# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Create sample Employee dataset
data = {
    "Age": [22, 35, 45, 28, 50, 41, 30, 26, 48, 33],
    "Salary": [20000, 50000, 70000, 30000, 90000, 65000, 40000, 25000, 85000, 48000],
    "YearsAtCompany": [1, 7, 15, 2, 20, 10, 4, 1, 18, 6],
    "JobSatisfaction": [2, 4, 3, 2, 5, 4, 3, 1, 5, 3],
    "Churn": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)

# Step 2: Split features and target
X = df[["Age", "Salary", "YearsAtCompany", "JobSatisfaction"]]
y = df["Churn"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Create Decision Tree Classifier (Entropy)
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Plot the Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Age", "Salary", "YearsAtCompany", "JobSatisfaction"],
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree Classifier for Employee Churn Prediction")
plt.show()

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Yaswanth R
RegisterNumber:  25007390
*/
```

## Output:
<img width="932" height="677" alt="Screenshot 2026-02-13 143249" src="https://github.com/user-attachments/assets/0b090c7c-baae-452e-8a3e-0c9f5a9d6864" />

<img width="1188" height="773" alt="image" src="https://github.com/user-attachments/assets/3a0d5723-3e7a-4986-8499-157b2fad90c0" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
