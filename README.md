# Smart Document Understanding System (ML + NLP)

An end-to-end Machine Learning and Natural Language Processing project that analyzes raw text and predicts its sentiment (Positive / Negative). The system includes text preprocessing, feature engineering using TF-IDF, model training with classical ML algorithms, and a Streamlit web application for real-time predictions.

---

## 🚀 Features

- Text preprocessing (lowercasing, punctuation removal, tokenization, lemmatization)
- TF-IDF vectorization
- Machine Learning models:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Model comparison and evaluation
- Saved trained model and vectorizer
- Streamlit web app for user input and prediction

---

## 🧠 Tech Stack

- Python  
- Pandas, NumPy  
- NLTK  
- Scikit-learn  
- Joblib  
- Streamlit  

---

## 📂 Project Structure

.
├── app.py  
├── main.ipynb  
├── sentiment_model.joblib  
├── tfidf_vectorizer.joblib  
├── requirements.txt  
└── README.md  

---

## 📊 Dataset

IMDb Movie Reviews Dataset  
Contains 50,000 movie reviews labeled as positive or negative.

Source: Kaggle (IMDb Dataset of 50K Movie Reviews)

---

## ⚙️ Installation

Install required libraries:

pip install -r requirements.txt

---

## ▶️ Run Application

streamlit run app.py

---

## 🧪 Example Usage

Input:  
The movie was slow but the acting was excellent.

Output:  
Sentiment: Positive

---

## 📈 Model Performance (Approximate)

- Naive Bayes: ~85% accuracy  
- Logistic Regression: ~88% accuracy  
- SVM: ~89–90% accuracy  

---

## 🏗 Workflow

1. Load dataset  
2. Perform EDA  
3. Clean and preprocess text  
4. Convert text to numerical features using TF-IDF  
5. Train multiple ML models  
6. Evaluate models  
7. Save best model  
8. Deploy using Streamlit  

---

## 🎯 Learning Outcomes

- Understand NLP preprocessing pipeline  
- Learn feature engineering for text  
- Compare ML algorithms for NLP  
- Build end-to-end ML system  
- Create simple ML web app  

---

## 🔮 Future Improvements

- Add topic classification  
- Add deep learning (LSTM / GRU)  
- Add confidence score  
- Improve UI  
- Add model explainability  

---

## 👤 Author

Prajwal Poojary
