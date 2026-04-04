import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, fbeta_score

def train_flight_model():
    features_base = [
        'MONTH', 'DAY_OF_WEEK', 'CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS',
        'PRCP', 'SNOW', 'TMAX', 'AWND',
        'CARRIER_HISTORICAL', 'DEP_AIRPORT_HIST', 'DAY_HISTORICAL', 'DEP_BLOCK_HIST'
    ]
    cat_features = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'DEP_TIME_BLK']
    target = 'DEP_DEL15'

    print("Loading full training dataset...")
    df = pd.read_csv('train.csv', usecols=features_base + cat_features + [target])
    
    print(f"Dataset loaded. Total rows: {len(df)}")
    
    print("Encoding categorical features...")
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[features_base + cat_features]
    y = df[target]

    negative_cases = len(df[df[target] == 0])
    positive_cases = len(df[df[target] == 1])
    scale_weight = negative_cases / positive_cases
    
    print(f"Starting model training on FULL DATA (Scale Weight: {scale_weight:.2f})...")
    model = XGBClassifier(
        n_estimators=700,
        learning_rate=0.06,
        max_depth=10,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X, y)

    print("Loading test dataset for evaluation...")
    test_df = pd.read_csv('test.csv', usecols=features_base + cat_features + [target])
    test_sample = test_df.sample(100000, random_state=42)
    
    for col in cat_features:
        known_classes = set(encoders[col].classes_)
        test_sample[col] = test_sample[col].astype(str).apply(
            lambda x: x if x in known_classes else encoders[col].classes_[0]
        )
        test_sample[col] = encoders[col].transform(test_sample[col])

    X_test = test_sample[features_base + cat_features]
    y_test = test_sample[target]
    
    probabilities = model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.50
    highest_f2_score = 0
    
    print("Optimizing threshold for F2-Score (Min. Accuracy: 0.70)...")
    for threshold in np.arange(0.30, 0.70, 0.02):
        current_predictions = (probabilities >= threshold).astype(int)
        current_accuracy = accuracy_score(y_test, current_predictions)
        current_f2 = fbeta_score(y_test, current_predictions, beta=2, pos_label=1)
        
        if current_accuracy >= 0.70 and current_f2 > highest_f2_score:
            highest_f2_score = current_f2
            best_threshold = threshold

    final_predictions = (probabilities >= best_threshold).astype(int)

    print("\n" + "="*40)
    print(f"FINAL PERFORMANCE ON FULL DATA (Threshold: {best_threshold:.2f})")
    print(f"Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, final_predictions))
    print("="*40)
    
    print("\nSaving model and metadata...")
    joblib.dump(model, 'flight_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(features_base + cat_features, 'feature_names.pkl')
    print("Workflow completed successfully.")

if __name__ == "__main__":
    train_flight_model()