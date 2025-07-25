import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.set_page_config(page_title="SpamShield", layout="centered")
st.title("📩 SpamShield - SMS Spam Detection")
st.markdown("Enter one or more messages (one per line) to check if they're **Spam** or **Not**.")

# Input
input_text = st.text_area("Enter your message(s) below:", height=250)

if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter at least one message.")
    else:
        messages = [msg.strip() for msg in input_text.split("\n") if msg.strip()]
        transformed_msgs = vectorizer.transform(messages)
        predictions = model.predict(transformed_msgs)
        confidences = model.predict_proba(transformed_msgs).max(axis=1) * 100

        st.markdown("## 🧾 Results:")
        for i, (msg, label, confidence) in enumerate(zip(messages, predictions, confidences)):
            label_display = (
                "<span style='color:red;font-weight:bold;'>🚨 Spam</span>" if label == 1
                else "<span style='color:green;font-weight:bold;'>✅ Not Spam</span>"
            )

            st.markdown(f"""
            <div style='border:1px solid #ddd; padding:10px; border-radius:10px; margin-bottom:10px'>
            📨 <b>Message {i+1}:</b><br><br>
            {msg}<br><br>
            <b>Prediction:</b> {label_display}<br>
            <b>Confidence:</b> {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

        # Pie chart summary
        st.markdown("### 📊 Spam vs Not Spam Summary")
        spam_count = sum(1 for label in predictions if label == 1)
        not_spam_count = len(predictions) - spam_count

        summary_df = pd.DataFrame({
            'Type': ['Spam', 'Not Spam'],
            'Count': [spam_count, not_spam_count]
        })

        fig = px.pie(summary_df, names='Type', values='Count',
                     color_discrete_map={'Spam': 'red', 'Not Spam': 'green'},
                     title='📈 Message Type Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Average confidence
        avg_confidence_spam = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == 1]) if spam_count else 0
        avg_confidence_not_spam = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == 0]) if not_spam_count else 0

        st.markdown(f"""
        ### 📌 Average Confidence:
        - 🚨 **Spam**: {avg_confidence_spam:.2f}%
        - ✅ **Not Spam**: {avg_confidence_not_spam:.2f}%
        """)

        # Footer
        st.markdown("🔒 _SpamShield does not store your messages. Your privacy is protected._")
