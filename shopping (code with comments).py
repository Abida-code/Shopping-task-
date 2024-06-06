# Import necessary libraries
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the proportion of the dataset to include in the test split
TEST_SIZE = 0.4

def main():
    # Load data from the CSV file
    evidence, labels = load_data("shopping.csv")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train the k-nearest neighbors model
    model = train_model(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Evaluate the model's performance
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print out the evaluation metrics
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1-score: {100 * f1:.2f}%")

def load_data(filename):
    # Map month names to numbers for easier processing
    month_to_num = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    evidence = []
    labels = []

    # Open and read the CSV file
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract relevant features and convert to appropriate data types
            row_evidence = [
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_to_num[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ]
            evidence.append(row_evidence)
            # Extract the label and convert to binary (1 if "TRUE", 0 if "FALSE")
            labels.append(1 if row["Revenue"] == "TRUE" else 0)
    
    return evidence, labels

def train_model(evidence, labels):
    # Initialize the k-nearest neighbors classifier with k=1
    model = KNeighborsClassifier(n_neighbors=1)
    # Train the model using the training data
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    
    return accuracy, precision, recall, f1

# Ensure the main function runs when the script is executed
if __name__ == "__main__":
    main()
