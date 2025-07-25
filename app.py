import streamlit as st
import joblib

# Load trained model
model = joblib.load("spam_model.pkl")

st.title("📩 SpamShield - SMS Spam Detection")
st.write("Enter a message to check if it's Spam or Not.")

# Input field
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0][1]

        if prediction == 1:
            st.error(f"🚨 Spam Detected!\n\nConfidence: {probability*100:.2f}%")
        else:
            st.success(f"✅ Not Spam.\n\nConfidence: {(1 - probability)*100:.2f}%")
