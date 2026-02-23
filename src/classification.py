import os
import csv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_phishing_csv(filepath):
    """
    Load phishing dataset from CSV file.

    Returns:
        header (list), rows (list of list)
    """
    if not os.path.exists(filepath):
        print(f"{filepath} not found.")
        return None, None

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows

def preprocess_phishing_data(header, rows):
    """
    Preprocess phishing dataset:
        - Remove index column and target column
        - Convert features and labels to int

    Returns:
        X: features list
        y: labels list
        feature_names: list of feature column names
    """

    # Identify index and target columns
    index_col = header.index(header[0])
    target_col = header.index("Result")

    X = []
    y = []
    for row in rows:
        features = []
        for i, value in enumerate(row):
            if i == index_col or i == target_col:
                continue
            features.append(int(value))
        label = int(row[target_col])
        X.append(features)
        y.append(label)

        feature_names = [
        name for i, name in enumerate(header) if i != index_col and i != target_col
        ]
    return X, y, feature_names

def print_classification_report(y_test, y_pred):
    """
    Print classification metrics
    """
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

def train_classification_model(csv_path="phishing_dataset.csv", model_choice="KNN", model_path=None):
    """
    Train classification model (KNN, SVM, DecisionTree) on phishing dataset.

    Parameters:
        csv_path (str): CSV file path
        model_choice (str): 'KNN', 'SVM', or 'DecisionTree'
        model_path (str): Path to save trained model
    """

    if not os.path.exists(csv_path):
        print("Phishing dataset CSV not found.")
        return

    print("Loading and preprocessing phishing dataset...")

    header, rows = load_phishing_csv(csv_path)
    if header is None:
        return
    
    X, y, feature_names = preprocess_phishing_data(header, rows)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Initialize model and scaler flag
    if model_choice == "KNN":
        k = int(input("Enter k ( number of neighbors ): "))
        model = KNeighborsClassifier(n_neighbors=k)
        use_scaler = True
        model_path = model_path or "models/classification_phishing_knn.pkl"
    elif model_choice == "SVM":
        C = float(input("Enter C ( regularization strength ): "))
        model = SVC(C=C, kernel="rbf", random_state=42)
        use_scaler = True
        model_path = model_path or "models/classification_phishing_svm.pkl"
    elif model_choice == "DecisionTree":
        depth = int(input("Enter max depth: "))
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        use_scaler = False
        model_path = model_path or "models/classification_phishing_dt.pkl"
    else:
        print("Invalid selection. Please try again.")
        return

    # Ensure directory exists
    dir_name = os.path.dirname(model_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Build pipeline with optional scaling
    if use_scaler:
        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    else:
        pipeline = Pipeline(steps=[
            ("model", model),
        ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print_classification_report(y_test, y_pred)
    
    # Save pipeline
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f) 
    print(f"Model saved to {model_path}")

def use_classification_model(csv_path="phishing_dataset.csv"):
    """
    Load saved classification model and predict using user input.
    """

    print("Choose classifier model to use:")
    print("1) KNN")
    print("2) SVM")
    print("3) Decision Tree")

    choice = input("> ")

    model_map = {"1": "models/classification_phishing_knn.pkl",
                 "2": "models/classification_phishing_svm.pkl",
                 "3": "models/classification_phishing_dt.pkl"}

    model_path = model_map.get(choice)
    
    if model_path is None or not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    print(f"Loading model: {model_path}")

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    filepath = "phishing_dataset.csv"

    if not os.path.exists(filepath):
        print("Phishing dataset CSV not found.")
        return
    
    header, rows = load_phishing_csv(filepath)
    if header is None:
        return
    
    X, y, feature_names = preprocess_phishing_data(header, rows)

    user_input = []
    print("\nEnter values for each feature: ")
    for name in feature_names:
        while True:
            try:
                value = int(input(f"{name}: "))
                user_input.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    prediction = pipeline.predict([user_input])
    label = "phishing" if prediction[0] == 1 else "legitimate"
    print(f"\nPrediction: {label}")

