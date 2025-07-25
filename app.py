import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("spam_model.pkl")

# Streamlit UI
st.set_page_config(page_title="📩 SpamShield - SMS Spam Detection", layout="centered")
st.title("📩 SpamShield - SMS Spam Detection")
st.markdown("Enter one or more messages (one per line) to check if they're Spam or Not.")

# Multi-message input
user_input = st.text_area("Enter your message(s) below:", height=200, placeholder="Type or paste multiple SMS messages...")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter at least one message.")
    else:
        messages = [msg.strip() for msg in user_input.split("\n") if msg.strip()]
        predictions = model.predict(messages)
        probabilities = model.predict_proba(messages)

        # Display results
        st.subheader("🧾 Results:")
        results_df = pd.DataFrame({
            "Message": messages,
            "Prediction": predictions,
            "Confidence": [f"{round(max(prob) * 100, 2)}%" for prob in probabilities],
            "Label": ["🚨 Spam" if pred == 1 else "✅ Not Spam" for pred in predictions]
        })

        for i, row in results_df.iterrows():
            st.markdown(f"""
            **📨 Message {i+1}:**
            > {row['Message']}

            **Prediction:** {row['Label']}  
            **Confidence:** {row['Confidence']}  
            ---
            """)

# Footer
st.markdown("---")
st.markdown("🔒 SpamShield does not store your messages. Your privacy is protected.")
