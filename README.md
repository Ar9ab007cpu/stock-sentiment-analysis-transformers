# Stock Sentiment Analysis using Transformer Models

This project develops an advanced Natural Language Processing pipeline to analyze social media sentiment related to financial markets. By combining traditional NLP techniques with transformer-based architectures, the system extracts investor sentiment signals from stock-related tweets.

---

## Project Overview

Investor sentiment plays a significant role in financial markets, often influencing short-term price movements and volatility. Social media platforms provide large volumes of unstructured textual data that can be leveraged to understand public perception toward specific stocks.

This project focuses on transforming raw tweet data into actionable sentiment insights using both classical NLP methods and state-of-the-art transformer models.

### Workflow
- Tweet data preprocessing and cleaning  
- Tokenization and stopword removal  
- Stemming and lemmatization  
- Sentiment scoring using TextBlob and VADER  
- Feature extraction using Bag-of-Words and TF-IDF  
- Transformer-based sentiment inference (BERT and RoBERTa)  
- Model persistence for future predictions  

---

## Dataset

The dataset consists of stock-related tweets filtered for a specific ticker.

### Processed Dataset Sample

| Date | Cleaned_Tweet | Sentiment_Label | Polarity | VADER_Sentiment |
|------|--------------|----------------|-----------|-----------------|
| 2022-09-29 | mainstream media has done an amazing job... | Positive | 0.60 | 0.0772 |
| 2022-09-29 | tesla delivery estimates are at around... | Neutral | 0.00 | 0.0000 |
| 2022-09-29 | hahaha why are you still trying to stop... | Negative | 0.06 | -0.7096 |

---

## Models and Techniques

### Traditional NLP
- TextBlob sentiment scoring  
- VADER sentiment analysis  
- Bag-of-Words  
- TF-IDF  

### Transformer Models
- BERT  
- RoBERTa  

Transformer architectures provide contextual language understanding and significantly outperform traditional approaches in many sentiment analysis tasks.

---

## Key Insights

- Social media contains measurable sentiment signals relevant to financial markets.
- Transformer models enable deeper contextual interpretation compared to lexicon-based techniques.
- Combining classical NLP with deep learning creates a robust sentiment analysis pipeline.

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- NLTK  
- Scikit-learn  
- Transformers (Hugging Face)  
- TensorFlow  
- TextBlob  
- Matplotlib  
- WordCloud  

---

## How to Run the Project

### Clone the repository
```bash
git clone https://github.com/Ar9ab007cpu/stock-sentiment-analysis-transformers.git
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download NLTK resources
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

### Run the notebook
```bash
jupyter notebook
```

Open:

```
Stock_Sentiment_Transformers.ipynb
```

---

## Project Highlights

- Built a transformer-powered sentiment analysis system  
- Combined classical NLP with deep learning models  
- Extracted investor sentiment from unstructured text  
- Engineered a reusable inference pipeline  
- Saved model weights for deployment scenarios  

---

## Future Improvements

- Fine-tuning transformers on financial text corpora  
- Real-time tweet streaming and sentiment scoring  
- Sentiment-driven trading signal research  
- Deployment as an API  
- Integration with market prediction models  

---
