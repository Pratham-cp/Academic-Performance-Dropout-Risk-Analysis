import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

st.set_page_config(page_title="Academic AI System", layout="wide")



@st.cache_data
def load_data():
    df = pd.read_csv("student-mat.csv", sep=';')
    df['internet'] = df['internet'].map({'yes': 1, 'no': 0})
    return df

df = load_data()

# ==========================
# Load Models
# ==========================

@st.cache_resource
def load_models():
    with open("grade_model.pkl", "rb") as f:
        grade_model = pickle.load(f)

    with open("risk_model.pkl", "rb") as f:
        risk_model = pickle.load(f)

    return grade_model, risk_model

grade_model, risk_model = load_models()

# ==========================
# TITLE
# ==========================

st.title("🎓 Academic Performance & Dropout Risk AI System")

# ==============================================================
# 1️⃣ DATA PREVIEW
# ==============================================================

st.header("📂 Data Preview")

st.write("Dataset Shape:", df.shape)
st.dataframe(df.head(10))

# ==============================================================
# 2️⃣ SUBJECT-WISE CORRELATION HEATMAP
# ==============================================================

st.header("📊 Subject-wise Grade Correlation Heatmap")

fig1, ax1 = plt.subplots()
plt.figure(figsize=(1,1))
sns.heatmap(df[['G1','G2','G3']].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax1)
st.pyplot(fig1)

# ==============================================================
# 3️⃣ ATTENDANCE VS FINAL GRADE
# ==============================================================

st.header("📍 Attendance vs Final Grade")

fig2, ax2 = plt.subplots()
plt.figure(figsize=(5,4))
sns.regplot(x='absences', y='G3', data=df, ax=ax2)
st.pyplot(fig2)

# ==============================================================
# 4️⃣ PROBABILITY OF PASSING
# ==============================================================

st.header("🎯 Probability of Passing")

df['pass'] = df['G3'] >= 10

pass_rate = df['pass'].mean() * 100

st.write(f"Overall Pass Percentage: {pass_rate:.2f}%")

fig3, ax3 = plt.subplots()
plt.figure(figsize=(5,4))
sns.countplot(x='pass', data=df, ax=ax3)
st.pyplot(fig3)

# ==============================================================
# 5️⃣ G3 DISTRIBUTION: MALE VS FEMALE
# ==============================================================

st.header("👨‍🎓👩‍🎓 G3 Distribution: Male vs Female")

fig4, ax4 = plt.subplots()
plt.figure(figsize=(5,4))
sns.boxplot(x="sex", y="G3", data=df, ax=ax4)
st.pyplot(fig4)

# ==============================================================
# 6️⃣ RISK DISTRIBUTION
# ==============================================================

df['risk'] = 0
df.loc[df['G3'] < 12, 'risk'] = 1
df.loc[df['G3'] < 8, 'risk'] = 2

st.header("⚠ Risk Distribution")

fig5, ax5 = plt.subplots()
plt.figure(figsize=(5,4))
sns.countplot(x="risk", data=df, ax=ax5)
st.pyplot(fig5)

# ==============================================================
# 7️⃣ USER INPUT SECTION
# ==============================================================

st.header("🧠 Enter Student Details for Prediction")

col1, col2 = st.columns(2)

with col1:
    g1 = st.slider("G1", 0, 20, 10)
    g2 = st.slider("G2", 0, 20, 10)
    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Failures", 0, 3, 0)

with col2:
    absences = st.slider("Absences", 0, 50, 5)
    internet = st.selectbox("Internet Access", [0,1])
    freetime = st.slider("Free Time", 1, 5, 3)

input_df = pd.DataFrame([[
    g1, g2, studytime, failures,
    absences, internet, freetime
]], columns=[
    'G1','G2','studytime','failures',
    'absences','internet','freetime'
])

# ==============================================================
# 8️⃣ PREDICTION
# ==============================================================

if st.button("Predict"):

    grade_prediction = grade_model.predict(input_df)[0]
    risk_prediction = risk_model.predict(input_df)[0]

    st.subheader("Prediction Result")

    st.success(f"Predicted Final Grade: {grade_prediction:.2f}")

    if risk_prediction == 0:
        st.success("Low Risk")

    elif risk_prediction == 1:
        st.warning("Medium Risk")

    else:
        st.error("High Dropout Risk")

# ==============================================================
# 9️⃣ FEATURE IMPORTANCE
# ==============================================================

st.header("📌 Feature Importance")

importance = grade_model.feature_importances_

features = [
    'G1','G2','studytime','failures',
    'absences','internet','freetime'
]

fig6, ax6 = plt.subplots()
plt.figure(figsize=(6, 4))
sns.barplot(x=importance, y=features, ax=ax6)
st.pyplot(fig6)

# ==============================================================
# 🔟 R² SCORE (AT THE END)
# ==============================================================

st.header("📈 Model Performance")

# Calculate R² again for display

from sklearn.model_selection import train_test_split

X = df[features]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = grade_model.predict(X_test)

r2 = r2_score(y_test, y_pred)

st.success(f"Final R² Score of Grade Model: {r2:.3f}")
