import streamlit as st
import joblib
import pandas as pd
from io import StringIO

# Load model once
model = joblib.load('spam_model.pkl')

st.title("📧 SpamShield - Spam Message Classifier")

st.markdown("Enter one or more messages below, each on a separate line:")

messages_input = st.text_area("Enter messages here:")

if st.button("Check Spam"):
    if not messages_input.strip():
        st.warning("Please enter at least one message.")
    else:
        # Split input by lines and filter out empty lines
        messages = [msg.strip() for msg in messages_input.split('\n') if msg.strip()]
        
        # Predict all messages
        predictions = model.predict(messages)
        
        # Prepare results DataFrame
        df = pd.DataFrame({
            'Message': messages,
            'Prediction': ['Spam 🚨' if p == 1 else 'Not Spam ✅' for p in predictions]
        })
        
        st.markdown("### Results:")
        st.dataframe(df)
        
        # Provide CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='spamshield_results.csv',
            mime='text/csv'
        )
