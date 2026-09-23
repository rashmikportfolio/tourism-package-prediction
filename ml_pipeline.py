# ============================================================
# Tourism Package Prediction - Automated ML Pipeline
# ============================================================

import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib


# ============================================================
# 1. Load training and testing data from Hugging Face
# ============================================================

print("STEP 1: Loading data from Hugging Face...")

train_url = (
    "https://huggingface.co/datasets/"
    "rashmikportfolio/tourism-package-prediction/"
    "resolve/main/train.csv"
)

test_url = (
    "https://huggingface.co/datasets/"
    "rashmikportfolio/tourism-package-prediction/"
    "resolve/main/test.csv"
)

train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)


# ============================================================
# 2. Separate features and target
# ============================================================

print("\nSTEP 2: Preparing features and target...")

target = "ProdTaken"

X_train = train_data.drop(columns=[target])
y_train = train_data[target]

X_test = test_data.drop(columns=[target])
y_test = test_data[target]


# ============================================================
# 3. Define numerical and categorical features
# ============================================================

numerical_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]


# ============================================================
# 4. Preprocessing
# ============================================================

print("\nSTEP 3: Applying preprocessing...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 5. Transform training and testing data
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# ============================================================
# 6. Train final Gradient Boosting model
# ============================================================

print("\nSTEP 4: Training Gradient Boosting model...")

# These are the tuned parameters selected during experimentation
best_params = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_samples_split": 5,
    "n_estimators": 200
}
model = GradientBoostingClassifier(
    **best_params,
    random_state=42
)

model.fit(X_train_processed, y_train)

print("Model training completed.")


# ============================================================
# 7. Evaluate model
# ============================================================

print("\nSTEP 5: Evaluating model...")

predictions = model.predict(X_test_processed)
probabilities = model.predict_proba(X_test_processed)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# 8. Save the complete preprocessing + model pipeline
# ============================================================

print("\nSTEP 6: Saving model pipeline...")

from sklearn.pipeline import Pipeline

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Refit complete pipeline on raw training data
final_pipeline.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)

model_path = "model/tourism_package_pipeline.joblib"

joblib.dump(final_pipeline, model_path)

print(f"Model saved to: {model_path}")
print("\nML pipeline completed successfully.")


# ============================================================
# 9.. Upload trained model to Hugging Face Model Hub
# ============================================================

print("\nSTEP 9: Uploading model to Hugging Face...")

from huggingface_hub import HfApi

hf_token = os.environ.get("HF_TOKEN")

model_repo = "rashmikportfolio/tourism-package-prediction-model"

if hf_token:
    api = HfApi(token=hf_token)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="tourism_package_pipeline.joblib",
        repo_id=model_repo,
        repo_type="model"
    )

    print("Model uploaded successfully to Hugging Face.")

else:
    print("HF_TOKEN not found.")
    print("Model upload skipped during local execution.")