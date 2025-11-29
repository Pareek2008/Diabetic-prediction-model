import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data_preprocessing import preprocess_data
from src.utils import save_model

def train(data_path, model_dir):

    print("\nLoading & preprocessing dataset...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data_path)

    print("Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model Accuracy: {acc:.4f}")

    print("\nSaving model...")
    save_model(model, scaler, model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to diabetes.csv")
    parser.add_argument("--model_dir", required=True, help="Directory to save model")

    args = parser.parse_args()
    train(args.data_path, args.model_dir)
