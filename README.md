# 🧠 Sentiment Analysis on Mental Health-related Reddit Posts

## 📌 Project Overview
This project aims to perform **sentiment analysis** on posts from mental health-related Reddit communities such as `r/depression`, `r/anxiety`, and `r/mentalhealth`.  
By leveraging Natural Language Processing (NLP) techniques, the goal is to classify the emotional tone of these posts and identify trends in online mental health discussions.

⚠ **Ethical Note:**  
This project is intended for **research and learning purposes only**.  
It does **not** attempt to diagnose or provide mental health advice.

---

## 🎯 Objectives
- Collect mental health-related posts from Reddit.
- Clean and preprocess text for analysis.
- Apply sentiment classification using:
  - VADER
  - TextBlob
  - Transformer-based models (e.g., BERT)
- Perform exploratory data analysis (EDA) to uncover insights.
- Visualize sentiment trends over time.

---

## 📂 Project Structure
reddit-mental-health-sentiment/
│
├── data/ # Raw and cleaned datasets
├── notebooks/ # Jupyter Notebooks for EDA & modeling
├── src/ # Python scripts for scraping, preprocessing, modeling
│ ├── fetch_data.py
│ ├── preprocess.py
│ └── sentiment_analysis.py
│
├── .gitignore
├── README.md
├── requirements.txt


---

## 🛠 Tech Stack
- **Language:** Python 3.9+
- **Libraries:**
  - Data handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - NLP: `nltk`, `spacy`, `vaderSentiment`, `textblob`, `transformers`
  - Reddit API: `praw`, `psaw`
  - Utility: `tqdm`, `beautifulsoup4`
- **Optional:** `streamlit` for an interactive dashboard

---

## 📊 Methodology
1. **Data Collection**  
   - Using **PRAW** or **Pushshift API (PSAW)** to scrape Reddit posts.
   - Targeting specific subreddits focused on mental health.

2. **Data Preprocessing**  
   - Removing stopwords, punctuation, URLs, and emojis.
   - Tokenization, lemmatization, and lowercasing.

3. **Sentiment Analysis**  
   - Applying VADER and TextBlob for polarity scoring.
   - Experimenting with BERT for deep learning classification.

4. **EDA & Visualization**  
   - Sentiment distribution plots.
   - Word clouds for positive vs negative posts.
   - Trend analysis over time.

5. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

---

## ⚙️ Installation
### 1️⃣ Clone the Repository
git clone https://github.com/<your-username>/reddit-mental-health-sentiment.git
cd reddit-mental-health-sentiment

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install Additional NLP Models
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# spaCy model
python -m spacy download en_core_web_sm


**🚀 Usage**
***Fetch Data**
Copy
Edit
python src/fetch_data.py

***Preprocess Data**
Copy
Edit
python src/preprocess.py

***Run Sentiment Analysis**
Copy
Edit
python src/sentiment_analysis.py

***View Results in Notebook**
Copy
Edit
jupyter notebook notebooks/EDA.ipynb

**📈 Sample Visualizations**


**📜 License**
This project is open-source under the MIT License.

**🙏 Acknowledgements**
Reddit API for data access
PRAW and PSAW libraries
VADER Sentiment Analysis
TextBlob
HuggingFace Transformers

