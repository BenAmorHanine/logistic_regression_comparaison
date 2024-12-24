import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to generate synthetic data
def generate_data():
    np.random.seed(0)
    x = np.random.rand(100) * 10  # Random data between 0 and 10
    y = (x > 5).astype(int)       # Binary target: 1 if x > 5, else 0
    return train_test_split(x, y, test_size=0.5, random_state=42)

# Gradient Descent Implementation
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m = len(y)
    X = np.c_[np.ones(X.shape[0]), X]  # Adding bias term
    beta = np.zeros(X.shape[1])        # Initializing coefficients

    for epoch in range(epochs):
        z = np.dot(X, beta)
        predictions = sigmoid(z)
        errors = y - predictions
        gradient = np.dot(X.T, errors) / m
        beta += lr * gradient

    return beta

# Newton's Method Implementation
def newton_method(X, y, epochs=10):
    m = len(y)
    X = np.c_[np.ones(X.shape[0]), X]  # Adding bias term
    beta = np.zeros(X.shape[1])        # Initializing coefficients

    for epoch in range(epochs):
        z = np.dot(X, beta)
        predictions = sigmoid(z)
        errors = predictions - y

        gradient = np.dot(X.T, errors) / m
        W = np.diag(predictions * (1 - predictions))
        hessian = np.dot(X.T, np.dot(W, X)) / m

        beta -= np.linalg.inv(hessian).dot(gradient)

    return beta

# Model Evaluation
def evaluate_model(beta, X_test, y_test):
    X_test_with_bias = np.c_[np.ones(X_test.shape[0]), X_test]  # Adding bias term
    y_pred_prob = sigmoid(np.dot(X_test_with_bias, beta))
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    return accuracy, precision, recall, auc

# Comparison with Sklearn's Logistic Regression
def compare_with_sklearn(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train.reshape(-1, 1), y_train)

    y_sklearn_pred = model.predict(X_test.reshape(-1, 1))
    accuracy = accuracy_score(y_test, y_sklearn_pred)
    return model.intercept_, model.coef_, accuracy

# Main function to run and compare both methods
def main():
    X_train, X_test, y_train, y_test = generate_data()

    # Gradient Descent
    beta_gd = gradient_descent(X_train, y_train, lr=0.1, epochs=1000)
    gd_metrics = evaluate_model(beta_gd, X_test, y_test)

    # Newton's Method
    beta_nm = newton_method(X_train, y_train, epochs=10)
    nm_metrics = evaluate_model(beta_nm, X_test, y_test)

    # Sklearn Logistic Regression
    sklearn_intercept, sklearn_coef, sklearn_accuracy = compare_with_sklearn(X_train, y_train, X_test, y_test)

    # Print Results
    print("Gradient Descent Coefficients (β0, β1):", beta_gd)
    print("Gradient Descent Metrics (Accuracy, Precision, Recall, AUC):", gd_metrics)

    print("Newton's Method Coefficients (β0, β1):", beta_nm)
    print("Newton's Method Metrics (Accuracy, Precision, Recall, AUC):", nm_metrics)

    print("Sklearn Coefficients (Intercept, Coef):", sklearn_intercept, sklearn_coef)
    print("Sklearn Accuracy:", sklearn_accuracy)

if __name__ == "__main__":
    main()
