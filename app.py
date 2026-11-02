
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path

# =========================
# Page setup
# =========================

st.set_page_config(
    page_title="Phishing Email Detection System",
    page_icon="📧",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_phishing_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
DATASET_PATH = BASE_DIR / "data" / "cleaned_phishing_email_dataset.csv"
RESULTS_PATH = BASE_DIR / "results" / "model_comparison_results.csv"
CM_PATH = BASE_DIR / "results" / "svm_confusion_matrix.csv"
# =========================
# Professional CSS design
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #061421 0%, #0b2440 45%, #11395f 100%);
        color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030b14 0%, #09223a 100%);
        border-right: 1px solid rgba(255,255,255,0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #ffffff !important;
    }

    .main-card {
        background: rgba(255,255,255,0.11);
        padding: 24px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 10px 30px rgba(0,0,0,0.28);
        margin-bottom: 18px;
    }

    .small-card {
        background: rgba(255,255,255,0.13);
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 15px;
    }

    .safe-box {
        background: rgba(0, 165, 90, 0.24);
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #00d084;
        margin-bottom: 15px;
    }

    .danger-box {
        background: rgba(210, 40, 60, 0.25);
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #ff4b5c;
        margin-bottom: 15px;
    }

    .warning-box {
        background: rgba(230, 170, 30, 0.25);
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #ffcc33;
        margin-bottom: 15px;
    }

    .info-box {
        background: rgba(0, 145, 255, 0.20);
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #00aaff;
        margin-bottom: 15px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.3rem;
        font-weight: 700;
    }

    .stDownloadButton > button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.13);
        padding: 15px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.16);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: white !important;
    }

    textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px !important;
    }

    input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    .stDataFrame {
        background-color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Cached loading for speed
# =========================

@st.cache_resource
def load_model_and_vectorizer():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


@st.cache_data
def load_dataset_and_results():
    df = pd.read_csv(DATASET_PATH)
    results_df = pd.read_csv(RESULTS_PATH)

    if CM_PATH.exists():
        cm_df = pd.read_csv(CM_PATH, index_col=0)
    else:
        cm_df = None

    return df, results_df, cm_df


required_files = {
    "Model": MODEL_PATH,
    "Vectorizer": VECTORIZER_PATH,
    "Dataset": DATASET_PATH,
    "Results": RESULTS_PATH
}

missing = []
for name, path in required_files.items():
    if not path.exists():
        missing.append(f"{name} missing: {path}")

if missing:
    st.error("Required files are missing.")
    for item in missing:
        st.write(item)
    st.stop()

model, vectorizer = load_model_and_vectorizer()
df, results_df, cm_df = load_dataset_and_results()

# =========================
# Helper functions
# =========================

suspicious_words = [
    "urgent", "verify", "password", "account", "click", "bank",
    "security", "update", "login", "limited", "suspended",
    "confirm", "payment", "winner", "free", "blocked"
]

marketing_words = [
    "unsubscribe", "shop now", "collection", "commercial communications",
    "instagram", "facebook", "tiktok", "pinterest", "newsletter",
    "offer", "sale", "app store", "google play"
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_keywords(text):
    text_lower = str(text).lower()
    found = [word for word in suspicious_words if word in text_lower]
    return found


def identify_email_type(text):
    text_lower = str(text).lower()

    if any(word in text_lower for word in marketing_words):
        return "Marketing or promotional email"
    elif any(word in text_lower for word in ["bank", "payment", "card", "transaction"]):
        return "Banking or payment-related email"
    elif any(word in text_lower for word in ["password", "login", "account", "verify"]):
        return "Account or login-related email"
    elif any(word in text_lower for word in ["delivery", "parcel", "shipment"]):
        return "Delivery-related email"
    elif any(word in text_lower for word in ["job", "application", "university", "student"]):
        return "Work or study-related email"
    else:
        return "General email"


def get_prediction(email_text):
    cleaned = clean_text(email_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    if prediction == 1:
        return "Phishing Email", 1
    return "Safe Email", 0


def get_risk_level(prediction, keywords, email_type):
    if prediction == "Safe Email":
        return "Low Risk"

    if prediction == "Phishing Email" and email_type == "Marketing or promotional email":
        return "Review Needed - Possible False Positive"

    if len(keywords) >= 4:
        return "High Risk"
    elif len(keywords) >= 2:
        return "Medium Risk"
    else:
        return "Medium Risk"


def get_advice(prediction, risk_level):
    if prediction == "Safe Email":
        return "This email looks safe based on the model result. However, users should still check the sender address and links before taking action."

    if risk_level == "Review Needed - Possible False Positive":
        return "This email may be a genuine promotional email, but it contains words that are also common in phishing emails. The user should check the sender, links and official website before clicking."

    return "Do not click links, do not download attachments, and do not enter passwords. Check the sender address and open the official website directly."


def lime_predict_proba(texts):
    cleaned_texts = [clean_text(t) for t in texts]
    features = vectorizer.transform(cleaned_texts)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)

    scores = model.decision_function(features)
    scores = np.clip(scores, -50, 50)
    phishing_prob = 1 / (1 + np.exp(-scores))
    safe_prob = 1 - phishing_prob
    return np.vstack([safe_prob, phishing_prob]).T


def generate_lime_explanation(email_text, predicted_class_num, prediction_label):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(
        class_names=["Safe Email", "Phishing Email"]
    )

    explanation = explainer.explain_instance(
        email_text,
        lime_predict_proba,
        labels=[0, 1],
        num_features=10,
        num_samples=300
    )

    lime_df = pd.DataFrame(
        explanation.as_list(label=predicted_class_num),
        columns=["Word / Feature", "Influence"]
    )

    lime_df["Absolute Influence"] = lime_df["Influence"].abs()

    if prediction_label == "Phishing Email":
        lime_df["Direction"] = lime_df["Influence"].apply(
            lambda x: "Supports Phishing" if x > 0 else "Supports Safe"
        )
    else:
        lime_df["Direction"] = lime_df["Influence"].apply(
            lambda x: "Supports Safe" if x > 0 else "Supports Phishing"
        )

    lime_df = lime_df.sort_values("Absolute Influence", ascending=False)

    return lime_df


def render_card(title, body):
    st.markdown(
        f"""
        <div class="main-card">
        <h3>{title}</h3>
        <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Sidebar
# =========================

st.sidebar.title("📧 AI Phishing Detector")
st.sidebar.write("MSc Dissertation Prototype")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Email Checker",
        "Batch Checker",
        "Dataset Overview",
        "Model Performance",
        "Cybersecurity Awareness",
        "About Project"
    ]
)

# =========================
# Main header
# =========================

st.markdown(
    """
    <div class="main-card">
    <h1>Explainable AI-Based Phishing Email Detection System</h1>
    <p>This dashboard predicts whether an email is safe or phishing and explains the result using risk indicators, LIME model-level explanation and cybersecurity awareness advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Pages
# =========================

if page == "Home":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cleaned Dataset Rows", f"{len(df):,}")
    col2.metric("Models Compared", len(results_df))
    col3.metric("Final Model", "SVM")

    render_card(
        "System Purpose",
        "The aim of this project is to detect phishing emails using machine learning and support users with explanation and cybersecurity awareness advice."
    )

    render_card(
        "Main Pipeline",
        "Dataset collection → data cleaning → TF-IDF feature extraction → model training → SVM selection → Streamlit dashboard → prediction, LIME explanation and awareness advice."
    )

elif page == "Email Checker":
    st.header("Email Checker")

    email_text = st.text_area("Paste email text here:", height=220)

    check_button = st.button("Check Email")

    if check_button:
        if not email_text.strip():
            st.warning("Please enter email text first.")
        else:
            prediction, predicted_class_num = get_prediction(email_text)
            keywords = find_keywords(email_text)
            email_type = identify_email_type(email_text)
            risk_level = get_risk_level(prediction, keywords, email_type)
            advice = get_advice(prediction, risk_level)

            st.session_state["last_email_text"] = email_text
            st.session_state["last_prediction"] = prediction
            st.session_state["last_predicted_class_num"] = predicted_class_num
            st.session_state["last_keywords"] = keywords
            st.session_state["last_email_type"] = email_type
            st.session_state["last_risk_level"] = risk_level
            st.session_state["last_advice"] = advice

            try:
                with st.spinner("Generating LIME model-level explanation..."):
                    lime_df = generate_lime_explanation(
                        email_text,
                        predicted_class_num,
                        prediction
                    )
                st.session_state["last_lime_df"] = lime_df
                st.session_state["lime_error"] = None

            except ModuleNotFoundError:
                st.session_state["last_lime_df"] = None
                st.session_state["lime_error"] = "LIME is not installed. Please run this command in Anaconda Prompt: pip install lime"

            except Exception as e:
                st.session_state["last_lime_df"] = None
                st.session_state["lime_error"] = f"LIME explanation could not be generated: {e}"

    if "last_prediction" in st.session_state:
        prediction = st.session_state["last_prediction"]
        predicted_class_num = st.session_state["last_predicted_class_num"]
        keywords = st.session_state["last_keywords"]
        email_type = st.session_state["last_email_type"]
        risk_level = st.session_state["last_risk_level"]
        advice = st.session_state["last_advice"]

        if prediction == "Phishing Email":
            st.markdown(
                f"""
                <div class="danger-box">
                <h2>Prediction: Phishing Email</h2>
                <p><b>Risk Level:</b> {risk_level}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="safe-box">
                <h2>Prediction: Safe Email</h2>
                <p><b>Risk Level:</b> {risk_level}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("User-Friendly Explanation")

        st.markdown(
            f"""
            <div class="small-card">
            <p><b>Possible Email Type:</b> {email_type}</p>
            <p><b>Suspicious Keywords:</b> {", ".join(keywords) if keywords else "None found"}</p>
            <p><b>Reason Details:</b> The model checked the email using TF-IDF features and the saved Support Vector Machine model. The dashboard also checks risk-related words and possible email type to provide a simple user-level explanation.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Cybersecurity Awareness Advice")
        st.markdown(
            f"""
            <div class="warning-box">
            <p>{advice}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("LIME Model-Level Explanation")

        st.markdown(
            """
            <div class="info-box">
            <p><b>LIME meaning:</b> LIME explains one individual prediction by showing which words influenced the model decision.</p>
            <p><b>How to read the values:</b> Positive values support the predicted class. Negative values push away from the predicted class.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.get("lime_error"):
            st.error(st.session_state["lime_error"])

        elif st.session_state.get("last_lime_df") is not None:
            lime_df = st.session_state["last_lime_df"]

            st.write("LIME explanation table:")
            st.dataframe(
                lime_df[["Word / Feature", "Influence", "Direction"]],
                use_container_width=True
            )

            st.subheader("LIME Influence Bar Chart")
            chart_df = lime_df.set_index("Word / Feature")[["Influence"]]
            st.bar_chart(chart_df)

            supports_predicted = lime_df[lime_df["Influence"] > 0]["Word / Feature"].tolist()
            against_predicted = lime_df[lime_df["Influence"] < 0]["Word / Feature"].tolist()

            st.subheader("LIME Summary")

            st.markdown(
                f"""
                <div class="small-card">
                <p><b>Predicted class explained by LIME:</b> {prediction}</p>
                <p><b>Top words supporting the predicted class:</b> {", ".join(supports_predicted[:5]) if supports_predicted else "None"}</p>
                <p><b>Top words against the predicted class:</b> {", ".join(against_predicted[:5]) if against_predicted else "None"}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

elif page == "Batch Checker":
    st.header("Batch Email Checker")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded columns:", batch_df.columns.tolist())

        possible_columns = ["text", "email", "Email Text", "message", "body"]
        text_column = None

        for col in possible_columns:
            if col in batch_df.columns:
                text_column = col
                break

        if text_column is None:
            st.error("CSV must contain one text column such as text, email, Email Text, message or body.")
        else:
            with st.spinner("Checking emails..."):
                predictions = []
                risks = []
                email_types = []
                keywords_list = []

                for text in batch_df[text_column].astype(str):
                    pred, pred_num = get_prediction(text)
                    keywords = find_keywords(text)
                    email_type = identify_email_type(text)
                    risk = get_risk_level(pred, keywords, email_type)

                    predictions.append(pred)
                    risks.append(risk)
                    email_types.append(email_type)
                    keywords_list.append(", ".join(keywords) if keywords else "None")

                batch_df["Prediction"] = predictions
                batch_df["Risk Level"] = risks
                batch_df["Email Type"] = email_types
                batch_df["Suspicious Keywords"] = keywords_list

            st.dataframe(batch_df.head(100), use_container_width=True)

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results",
                csv,
                "batch_phishing_results.csv",
                "text/csv"
            )

elif page == "Dataset Overview":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Emails", f"{len(df):,}")

    if "label" in df.columns:
        col2.metric("Safe Emails", f"{(df['label'] == 'Safe Email').sum():,}")
        col3.metric("Phishing Emails", f"{(df['label'] == 'Phishing Email').sum():,}")
    else:
        col2.metric("Safe Emails", "N/A")
        col3.metric("Phishing Emails", "N/A")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Class Distribution")
    if "label" in df.columns:
        st.bar_chart(df["label"].value_counts())

elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Model Comparison Results")
    st.dataframe(results_df, use_container_width=True)

    metric_columns = ["Accuracy", "Precision", "Recall", "F1-score"]
    if all(col in results_df.columns for col in metric_columns):
        st.subheader("Performance Bar Chart")
        chart_df = results_df.set_index("Model")[metric_columns]
        st.bar_chart(chart_df)

    st.subheader("SVM Confusion Matrix")
    if cm_df is not None:
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.warning("Confusion matrix file not found.")

elif page == "Cybersecurity Awareness":
    st.header("Cybersecurity Awareness Advice")

    render_card(
        "Check the Sender",
        "Always check whether the sender email address matches the real organisation."
    )

    render_card(
        "Avoid Suspicious Links",
        "If an email asks you to click a link urgently, open the official website directly instead."
    )

    render_card(
        "Do Not Share Passwords",
        "Trusted organisations will not usually ask users to enter passwords through email links."
    )

    render_card(
        "Be Careful With Attachments",
        "Do not download unexpected attachments from unknown or suspicious senders."
    )

elif page == "About Project":
    st.header("About Project")

    render_card(
        "Dissertation Title",
        "An Explainable AI-Based Phishing Email Detection and Cybersecurity Awareness System."
    )

    render_card(
        "Technologies Used",
        "Python, Pandas, Scikit-learn, TF-IDF, Support Vector Machine, Streamlit and LIME."
    )

    render_card(
        "Explainability",
        "The system provides user-friendly explanation using risk level, suspicious keywords, possible email type, reason details and cybersecurity awareness advice. It also includes LIME model-level explanation with a table and bar chart for individual email predictions."
    )
