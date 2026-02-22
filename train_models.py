import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("student-mat.csv", sep=';')

# Preprocessing: Convert categorical to numeric
df['internet'] = df['internet'].map({'yes': 1, 'no': 0})

# Define common features used in both models
features = ['G1', 'G2', 'studytime', 'failures', 'absences', 'internet', 'freetime']
X = df[features]

# ---------------- 1. Grade Prediction Model (Regression) ----------------
y_grade = df['G3']
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y_grade, test_size=0.2, random_state=42)

grade_model = RandomForestRegressor(n_estimators=100, random_state=42)
grade_model.fit(X_train_g, y_train_g)

# Save Grade Model
with open("grade_model.pkl", "wb") as f:
    pickle.dump(grade_model, f)

# ---------------- 2. Risk Prediction Model (Classification) ----------------
# Logic: 0 = Low Risk (>=12), 1 = Medium Risk (8-11), 2 = High Risk (<8)
df['risk'] = 0 
df.loc[(df['G3'] >= 8) & (df['G3'] < 12), 'risk'] = 1
df.loc[df['G3'] < 8, 'risk'] = 2

y_risk = df['risk']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk, test_size=0.2, random_state=42)

risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train_r, y_risk[X_train_r.index]) # Ensure indices match

# Save Risk Model
with open("risk_model.pkl", "wb") as f:
    pickle.dump(risk_model, f)

print("Models trained and saved successfully!")