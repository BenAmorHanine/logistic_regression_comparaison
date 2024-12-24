import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Replace zeros with NaN in specific columns
    cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_to_check] = data[cols_to_check].replace(0, np.nan)
    
    # Impute missing values with column mean
    data.fillna(data.mean(), inplace=True)
    
    # Split into features and target
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.5, random_state=42)

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
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_sklearn_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_sklearn_pred)
    return model.intercept_, model.coef_, accuracy

# Function to plot results
def plot_results(metrics_gd, metrics_nm, sklearn_accuracy):
    labels = ['Accuracy', 'Precision', 'Recall', 'AUC']
    gd_values = metrics_gd
    nm_values = metrics_nm
    sklearn_values = [sklearn_accuracy, 0, 0, 0]  # Accuracy with placeholders for other metrics

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, gd_values, width, label='Gradient Descent')
    ax.bar(x, nm_values, width, label="Newton's Method")
    ax.bar(x + width, sklearn_values, width, label='Sklearn')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Logistic Regression Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Main function to run and compare methods
def main():
    file_path = './diabetes_dataset.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # Gradient Descent
    beta_gd = gradient_descent(X_train, y_train, lr=0.1, epochs=1000)
    gd_metrics = evaluate_model(beta_gd, X_test, y_test)

    # Newton's Method
    beta_nm = newton_method(X_train, y_train, epochs=10)
    nm_metrics = evaluate_model(beta_nm, X_test, y_test)

    # Sklearn Logistic Regression
    sklearn_intercept, sklearn_coef, sklearn_accuracy = compare_with_sklearn(X_train, y_train, X_test, y_test)

    # Print Results
    print("Gradient Descent Coefficients (β0, β1...):", beta_gd)
    print("Gradient Descent Metrics (Accuracy, Precision, Recall, AUC):", gd_metrics)

    print("Newton's Method Coefficients (β0, β1...):", beta_nm)
    print("Newton's Method Metrics (Accuracy, Precision, Recall, AUC):", nm_metrics)

    print("Sklearn Coefficients (Intercept, Coef):", sklearn_intercept, sklearn_coef)
    print("Sklearn Accuracy:", sklearn_accuracy)

    # Plot Results
    plot_results(gd_metrics, nm_metrics, sklearn_accuracy)

if __name__ == "__main__":
    main()
