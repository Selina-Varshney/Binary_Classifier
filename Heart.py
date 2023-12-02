
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve

# Load the data
heart_data = pd.read_csv('heart_disease_data.csv')

# Split the data into features (X) and target variable (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Dictionary to store accuracy values
accuracy_dict = {}

def evaluate_model(model, model_name):
    global accuracy_dict
    
    # Accuracy on training data
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    # Accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    # Save accuracy values in the dictionary
    accuracy_dict[model_name] = {'Training Accuracy': training_data_accuracy, 'Test Accuracy': test_data_accuracy}

    # Confusion Matrix
    conf_matrix = confusion_matrix(Y_test, X_test_prediction)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='{} - ROC curve (AUC = {:.2f})'.format(model_name, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(Y_test, model.predict_proba(X_test)[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'{model_name} - Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Input data for prediction
    input_data = (60, 1, 0, 125, 258, 0, 0, 141, 1, 2.8, 1, 1, 3)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make predictions
    prediction = model.predict(input_data_reshaped)

    # Display the result
    if prediction[0] == 0:
        print(f'{model_name} - The Person does not have Heart Disease')
    else:
        print(f'{model_name} - The Person has Heart Disease')

# Train the Decision Tree model
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, Y_train)
evaluate_model(model_dt, 'Decision Tree')

# Train the Random Forest model
model_rf = RandomForestClassifier()
model_rf.fit(X_train, Y_train)
evaluate_model(model_rf, 'Random Forest')

# Train the KNN model
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, Y_train)
evaluate_model(model_knn, 'KNN')

# Train the Logistic Regression model
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)
evaluate_model(model_lr, 'Logistic Regression')

# Find the algorithm with the highest test accuracy
best_model = max(accuracy_dict, key=lambda x: accuracy_dict[x]['Test Accuracy'])

print(f'\nThe best-performing algorithm is: {best_model}')
print('Accuracy values:')
print(f"Decision Tree - Training Accuracy: {accuracy_dict['Decision Tree']['Training Accuracy']}, Test Accuracy: {accuracy_dict['Decision Tree']['Test Accuracy']}")
print(f"Random Forest - Training Accuracy: {accuracy_dict['Random Forest']['Training Accuracy']}, Test Accuracy: {accuracy_dict['Random Forest']['Test Accuracy']}")
print(f"KNN - Training Accuracy: {accuracy_dict['KNN']['Training Accuracy']}, Test Accuracy: {accuracy_dict['KNN']['Test Accuracy']}")
print(f"Logistic Regression - Training Accuracy: {accuracy_dict['Logistic Regression']['Training Accuracy']}, Test Accuracy: {accuracy_dict['Logistic Regression']['Test Accuracy']}")
