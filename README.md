# 📩 SpamShield - SMS Spam Detection App

SpamShield is an AI-powered tool that detects spam messages using NLP and machine learning. Enter multiple SMS messages and instantly find out which ones are spam.

---

## 🧠 Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Model**: Multinomial Naive Bayes (Scikit-learn)
- **Visualization**: Plotly
- **Dataset**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🚀 Features
- Predicts whether messages are spam or not
- Shows confidence levels
- Summarizes results with charts
- Fast, offline model (no APIs needed)

---

## 📦 Installation

git clone https://github.com/aaryanpawar16/SpamShield.git
cd SpamShield/backend
pip install -r requirements.txt
streamlit run app.py

📊 Dataset
This project uses the SMS Spam Collection Dataset from UCI/Kaggle.

🏗️ Training the Model
cd SpamShield/training
python train_model.py
This will generate:

spam_model.pkl (the trained model)

vectorizer.pkl (the TF-IDF vectorizer)

💡 Deployment
To deploy on Streamlit Cloud:

Push this repo to GitHub

Go to https://streamlit.io/cloud

Link your GitHub repo

Set the main file as backend/app.py

Add environment variable PYTHON_VERSION=3.10

🙌 Credits
Dataset: UCI/Kaggle

Built with: Python, Scikit-learn, Streamlit

✨ Demo
Check out Demo Video on YouTube
https://youtu.be/-zSGI8kVX30?si=ERzUfDQUkTJJyt2w


##Check out project on my Streamlit App Link
https://spamshield11.streamlit.app/


