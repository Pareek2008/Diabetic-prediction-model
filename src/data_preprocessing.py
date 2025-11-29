import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    # Load dataset - detect separator
    try:
        # First try tab-separated (most common for this dataset)
        df = pd.read_csv(data_path, sep="\t")
        if "Glucose" not in df.columns:
            # If not found, try comma-separated
            df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # Replace zero values where zero is invalid
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid_cols:
        df[col] = df[col].replace(0, df[col].median())

    # Split input and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
