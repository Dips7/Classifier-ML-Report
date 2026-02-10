
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}

# List to store results
results = []

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("-" * 60)

    # Save results
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        "Model": name,
        "Accuracy": report['accuracy'],
        "Macro Avg Precision": report['macro avg']['precision'],
        "Macro Avg Recall": report['macro avg']['recall'],
        "Macro Avg F1-Score": report['macro avg']['f1-score'],
        "Weighted Avg Precision": report['weighted avg']['precision'],
        "Weighted Avg Recall": report['weighted avg']['recall'],
        "Weighted Avg F1-Score": report['weighted avg']['f1-score']
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('classification_results.csv', index=False)
print("\nResults saved to classification_results.csv")
