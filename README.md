

<!-- ===================== BIG MOVING TITLE ===================== -->
<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=56&duration=2500&pause=700&color=58A6FF&center=true&vCenter=true&width=1200&lines=Mental+Health+Monitoring" />
</p>

<!-- ===================== SMALL MOVING SUBTITLE ===================== -->
<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=24&duration=3000&pause=800&color=0A66C2&center=true&vCenter=true&width=1200&lines=Sentiment+Analysis+using+Machine+Learning+%7C+NLP+Research;Gold+Medalist+Final+Year+Thesis+Project;AI+for+Healthcare+and+Real+World+Impact" />
</p>



<p>
  <a><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"></a>
  <a><img src="https://img.shields.io/badge/Research-Thesis%20Project-6A5ACD?style=flat-square"></a>
  <a><img src="https://img.shields.io/badge/ML-NLP%20Based-0A66C2?style=flat-square"></a>
  <a href="https://drive.google.com/drive/folders/1YEB0w_tDlg8XT8nWkCjybzR0MWI3kf5E?usp=drive_link"><img src="https://img.shields.io/badge/Download-Dataset-success?style=flat-square&logo=google-drive" alt="Download Dataset"></a>
  <a href="https://www.linkedin.com/in/maazkareem-ai"><img src="https://img.shields.io/badge/LinkedIn-Maaz%20Kareem-blue?style=flat&logo=linkedin" alt="LinkedIn"></a>
  <a href="https://www.kaggle.com/maazkareem"><img src="https://img.shields.io/badge/Kaggle-maazkareem-brightgreen?style=flat&logo=kaggle&logoColor=white" alt="Kaggle"></a></a>
</p>


This repository contains the complete implementation of the **Final Year Research Thesis Project** titled  
**“Sentiment Analysis for Mental Health Monitoring using Machine Learning”**  
by **Maaz Kareem**, BS Computer Science (Gold Medalist), University of Buner, Pakistan (2025).

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
- **Purpose:** Cross-domain testing and model robustness testing  
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
| Logistic Regression | 93.64% | 93.63% | 93.63% |
| **SVM** | **94.16%** | **94.17%** | **94.17%** |
| Random Forest | 93.47% | 93.49% | 93.48% |
| LightGBM | 93.63% | 93.64% | 93.64% |

### Multi-Class Classification
| Model | Combined Data | Sentiment MH Data | Cross-Domain (Merged) |
|--------|----------------|-------------------|-----------------------|
| Logistic Regression | 74.83% | 39.41% | 52.68% |
| SVM | 75.95% | 40.24% | 56.64% |
| Random Forest | 74.53% | **49.11%** | **58.84%** |
| **LightGBM** | **77.65%** | 42.73% | 57.34% |

**Key Findings:**
- SVM consistently outperformed others with up to **94% accuracy** in binary classification.
- While LightGBM perform good on trained dataset havine **78% accuracy** but Random Forest perform good on new datasets having **49% and 59% accuracy** respectively
- Classical ML models proved interpretable and robust across domains.

---

## 🔍 Visualizations
- Confusion matrices (binary & multi-class)
- ROC curves
- Feature importance plots (Random Forest & LightGBM)
- F1-score bar charts
- classification report

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
## 🧾 Citation

If you use this repository for your research or publications, please cite it as:

> **Kareem, M. (2025).** *Sentiment Analysis for Mental Health Monitoring using Machine Learning.*  
> Undergraduate Research Thesis Project, Department of Computer Science, University of Buner, Pakistan.

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
```
---
## 👨‍💻 Author

**Maaz Kareem**  
🎓 *B.Sc. Computer Science (Gold Medalist)*  
🏫 *University of Buner, Pakistan*  

**Get in Touch:**  
I am passionate about Machine Learning, Data Science, and AI.  
For collaboration, questions, or professional inquiries, feel free to reach out via (**maaz.kareem.ai@gmail.com**).  
I welcome discussions, feedback, and knowledge sharing.
