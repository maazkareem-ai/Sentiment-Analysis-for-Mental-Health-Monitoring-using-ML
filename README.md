# 🧠 Mental Health Sentiment Analysis using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()

This repository contains the complete implementation of the **Final Year Research Thesis** titled  
**“Sentiment Analysis for Mental Health Monitoring using Machine Learning”**  
by **Maaz Karim**, BS Computer Science (Gold Medalist), University of Buner, Pakistan (2025).

---

## 📄 Project Overview

The objective of this research is to detect and classify mental health conditions from textual data collected from **social media platforms** using **Machine Learning (ML)** techniques.  
The focus is on classical ML algorithms for **interpretability, transparency, and efficiency** compared to deep learning approaches.

The system classifies posts into mental health categories such as:
- Normal  
- Depression  
- Anxiety  
- Stress  
- Suicidal Ideation  
- Bipolar Disorder  
- Personality Disorder  

Two main tasks were conducted:
1. **Binary Classification** – Normal vs. Abnormal (any mental health issue)  
2. **Multi-Class Classification** – Seven distinct mental health categories  

---

## 🧩 Datasets Used

This research utilized **three ethically sourced datasets** to ensure robustness and cross-domain generalization.

### 🧮 Dataset 1: Combined_Data.csv
- **Purpose:** Training, validation, and testing of ML models  
- **Source:** Aggregated from multiple Kaggle datasets including:  
  - 3k Conversations Dataset for Chatbot  
  - Depression Reddit Cleaned  
  - Human Stress Prediction  
  - Predicting Anxiety in Mental Health Data  
  - Students Anxiety and Depression Dataset  
  - Suicidal Tweet Detection Dataset  
- **Categories:** 7 (Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder)

### 🧮 Dataset 2: Sentiment_Mental_Health_Dataset.csv
- **Purpose:** Cross-domain validation and model robustness testing  
- **Sources:**  
  - *Zenodo Reddit Mental Health Dataset* (for 5 categories)  
  - *Reddit PRAW API* (for Stress & Personality Disorder posts)  
- **Ethics:** Only publicly available posts were used; usernames and PII were excluded.  
- **Categories:** Same seven as above  

### 🧮 Dataset 3: Final_Merged_Dataset_Cleaned.csv
- **Purpose:** Final cross-domain generalization testing  
- **Description:** Merged dataset combining *Sentiment_Mental_Health_Dataset.csv* with **20%** of data from *Combined_Data.csv*  

---

## ⚙️ Methodology

1. **Text Preprocessing**
   - Cleaning (removing URLs, symbols, and punctuation)
   - Tokenization and Lemmatization
   - Stop-word removal
   - Label encoding

2. **Feature Extraction**
   - TF-IDF vectorization (unigrams and bigrams)

3. **Model Training**
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - Random Forest  
   - Light Gradient Boosting Machine (LightGBM)

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC  
   - Cross-validation (k=5) for reliability  
   - Cross-domain testing to evaluate generalization

---

## 📈 Experimental Results

### Binary Classification
| Model | Combined Data | Sentiment MH Data | Cross-Domain (Merged) |
|--------|----------------|-------------------|-----------------------|
| Logistic Regression | 91.3% | 89.8% | 88.7% |
| SVM | 92.5% | 91.4% | 90.8% |
| Random Forest | 88.7% | 87.5% | 85.9% |
| **LightGBM** | **93.4%** | **92.6%** | **92.1%** |

### Multi-Class Classification
| Model | Combined Data | Sentiment MH Data | Cross-Domain (Merged) |
|--------|----------------|-------------------|-----------------------|
| Logistic Regression | 84.6% | 82.8% | 82.0% |
| SVM | 86.7% | 84.5% | 83.4% |
| Random Forest | 82.3% | 80.2% | 79.1% |
| **LightGBM** | **88.9%** | **86.1%** | **85.3%** |

**Key Findings:**
- LightGBM consistently outperformed others with up to **93% accuracy** in binary classification.
- Classical ML models proved interpretable and robust across domains.

---

## 🔍 Visualizations
- Confusion matrices (binary & multi-class)
- ROC curves
- Feature importance plots (Random Forest & LightGBM)
- F1-score bar charts

All visual outputs are stored in the `/results/` directory.

---

## 🧠 Key Contributions

- Development of an interpretable ML-based mental health classification system  
- Integration of **three multi-source datasets** for enhanced generalization  
- Validation of model stability through **cross-domain evaluation**  
- Demonstration of ML’s ethical and transparent use in psychological research  

---

## ⚖️ Ethical Considerations

- Data collected solely from **publicly available** sources (Reddit, Kaggle, Zenodo).  
- No private or identifiable user data were included.  
- Research complies with academic ethical standards for responsible AI use in mental health.

---

## 💡 Future Work

- Incorporate transformer-based embeddings (e.g., BERT, RoBERTa)  
- Add explainability with SHAP/LIME  
- Develop real-time mental health monitoring tools  
- Extend datasets to include multilingual data (Urdu, Pashto, etc.)

---

## 🧪 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/Mental-Health-Sentiment-Analysis-Using-Machine-Learning.git
cd Mental-Health-Sentiment-Analysis-Using-Machine-Learning

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/ML_Binary_Class.ipynb
jupyter notebook notebooks/ML_Multi_Class.ipynb
