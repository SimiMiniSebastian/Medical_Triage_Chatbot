import streamlit as st
import joblib
import random

# Load model and label encoder
model = joblib.load("models/chatbot_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# General post-diagnosis suggestions
suggestions = [
    "💧 Stay hydrated – drink plenty of water.",
    "🍽️ Do not skip meals; maintain a nutritious diet.",
    "🛌 Get enough rest and sleep.",
    "💊 Take supplements or medicines only if prescribed by a doctor.",
    "🧘‍♂️ Practice stress-relieving activities like meditation or yoga.",
    "📅 Schedule a follow-up with a healthcare provider."
]

# --- Page Setup ---
st.set_page_config(page_title="Medical Triage Chatbot", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: rgba(255, 255, 255, 0.92);
        padding: 30px;
        border-radius: 15px;
    }
    body {
        background-image: url("https://images.unsplash.com/photo-1588776814546-ec7d11878bce?ixlib=rb-4.0.3&auto=format&fit=crop&w=1650&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    h1, h2, h3, label, .stButton>button {
        color: #003366;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

# Title
st.title("🤖 Smart Medical Triage Assistant")

# Input: Demographic info
age = st.number_input("👤 Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("⚧ Gender", ["Prefer not to say", "Male", "Female", "Other"])
preexisting = st.text_input("🩺 Do you have any preexisting conditions or allergies?", placeholder="e.g. diabetes, penicillin allergy")

# Input: Symptoms
symptoms = st.text_input("📝 Enter your symptoms (comma-separated)", placeholder="e.g. fever, cough, headache")

if st.button("🔍 Diagnose"):
    if symptoms.strip():
        input_vector = [symptoms]
        prediction = model.predict(input_vector)
        disease = label_encoder.inverse_transform(prediction)[0]

        # Show diagnosis first
        st.subheader("🧾 Possible Disease(s):")
        st.success(f"Based on your symptoms, the possible disease is: **{disease}**")

        # Then show suggestions
        st.subheader("📋 General Health Suggestions:")
        for tip in random.sample(suggestions, 3):
            st.info(tip)

        # Reminder
        st.warning("⚠️ Please consult a medical practitioner for an accurate diagnosis.")

    else:
        st.warning("Please enter your symptoms.")

st.markdown("</div>", unsafe_allow_html=True)
