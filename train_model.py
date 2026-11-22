"""Train the same tuned Random Forest used in the Diabetes_Analyst notebook."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "diabetes_prediction_dataset.csv"
MODEL_PATH = ROOT / "model" / "diabetes_model.pkl"

# Load and clean the original dataset
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()

numeric_columns = df.select_dtypes(include="number").columns.drop("diabetes")
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

categorical_columns = df.select_dtypes(exclude="number").columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.dropna(subset=["diabetes"]).copy()
X = df.drop(columns=["diabetes"])
y = df["diabetes"].astype(int)

# Match the notebook's 80:20 stratified split and one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_encoded = pd.get_dummies(
    X_train, columns=["gender", "smoking_history"], dtype=int
)
X_test_encoded = pd.get_dummies(
    X_test, columns=["gender", "smoking_history"], dtype=int
).reindex(columns=X_train_encoded.columns, fill_value=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Best parameters recorded in Cell 12 of the notebook's GridSearchCV output
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
print(f"Data setelah cleaning: {len(df):,}")
print(f"Jumlah fitur: {len(X_train_encoded.columns)}")
print(f"Test Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"Test Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
print(f"Test Recall   : {recall_score(y_test, pred, zero_division=0):.4f}")
print(f"Test F1-Score : {f1_score(y_test, pred, zero_division=0):.4f}")

joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X_train_encoded.columns),
    },
    MODEL_PATH,
)
print(f"Model saved: {MODEL_PATH}")
