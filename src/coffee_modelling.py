
"""
lean utility module for DABN22 special project (Jia Yang Le, 20010428-T415)
"""
from __future__ import annotations
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

################
# Data wrangling
################
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drops ID andb redundant caffeine column
    - do some ordinal-encoding for inputs: sleep_quality, stress_level, health_issues
    - add BMI_category, age_group, high_caffeine_intake 
    """
    data = df.copy()        # so it does not mutate input

    data = data.drop(columns=[c for c in ['ID', 'Caffeine_mg'] if c in data.columns], errors='ignore')  # drop identified redundant columns

        # identify and perform ordinal encodings
    mappings = {
        'Sleep_Quality': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4},
        'Stress_Level': {'Low': 1, 'Medium': 2, 'High': 3},
        'Health_Issues': {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3},
    }
    
    for col, mapping in mappings.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
            if col == "Health_Issues":
                data[col] = data[col].fillna(0).astype(int)  # NaN treated as none or 0

    if 'BMI' in data.columns:
        data['BMI_Category'] = pd.cut(
            data['BMI'],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )                                    # BMI categorization, efficient

    if 'Age' in data.columns:
        data['Age_Group'] = pd.cut(
            data['Age'],
            bins=[17, 30, 50, 120],
            labels=['Young', 'Middle', 'Senior']
        )                                      # rough categorization of age groups, memory-efficient

    if 'High_Caffeine_Intake' not in data.columns:
        est_caffeine = None
        if 'Caffeine_mg' in df.columns:
            est_caffeine = df['Caffeine_mg']
        elif 'Coffee_Intake' in df.columns:
            est_caffeine = df['Coffee_Intake'] * 95.0
        if est_caffeine is not None:
            data['High_Caffeine_Intake'] = (est_caffeine > 400.0).astype(int)     # high caffeine flag (use Coffee_Intake * 95mg if Caffeine_mg missing)

    return data

##########################
# Feature matrix & target
##########################
def build_features_and_target(df: pd.DataFrame, target_col: str = 'Occupation') -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    - One-hot encoded nominal categoricals (Gender, Country, BMI_Category, Age_Group)
    - numeric/ordinal columns left as-is
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # One-hot encode common nominal columns if present
    nominal_cols = [c for c in ['Gender', 'Country', 'BMI_Category', 'Age_Group'] if c in X.columns]
    X = pd.get_dummies(X, columns=nominal_cols, drop_first=False)

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le

############################
# Train/test split & model
############################
def train_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split data and train a RandomForest classifier. Returns (model, X_train, X_test, y_train, y_test)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test
