# 🚀 Job & Resume Recommender System

A smart career assistant that matches **your resume** with **job datasets** to suggest:
- ✅ Relevant job roles you are a good fit for.  
- ✅ Missing skills & suitable projects to improve your chances.  

Built with **Python**, **pandas**, and **NLP-based parsing**.  

---

## 📌 Features
- **Resume Parsing** → extracts skills & projects automatically.  
- **Job Role Matching** → compares your skills to requirements in job datasets.  
- **Recommendations**:
  - Alternative job roles where your skills overlap better.  
  - Project ideas to add (if resume has <2 projects or missing required skills).  
- **Extensible Taxonomy** → `skills_taxonomy.json` for skill mappings & aliases.  
- **Datasets Supported**:
  - IT Job Dataset (Sri Lanka)  
  - Dice.com Jobs (US)  
  - Morocco Jobs Sample  

---


## 📂 Project Structure
```
AI-ML_Projects/
└── Capstone-Job-Recommender/
├── datasets/
│ ├── prepared/
│ ├── processed/
│ │ ├── jobs_merged.csv
│ │ └── skills_taxonomy.json
│ └── raw/
│
├── notebooks/
│ ├── EDA_dice_com.ipynb
│ ├── EDA_IT_Job.ipynb
│ ├── EDA_merge.ipynb
│ ├── EDA_morocco.ipynb
│ └── hidden.ipynb
│
├── .gitignore
├── README.md
├── requirements.txt
├── To-Do.md
└── Useful_Commands.md

````

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/job-resume-project.git
   cd job-resume-project
   ```

2. Create a virtual environment & install requirements:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / Mac
   venv\Scripts\activate      # Windows

   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Run in Jupyter

1. Open `notebooks/feature_engineering/1_setup.ipynb` to load datasets.
2. Process with `2_preprocessing.ipynb`.
3. Try recommendations in `3_recommender.ipynb`.

### Run demo app

```bash
streamlit run app.py
```

---

## 📊 Example Flow

* Input Resume:

  ```
  Skills: Python, Pandas
  Projects: 1 (Movie Recommender)
  Target Job: Data Scientist
  ```
* Output:

  * Suggests Data Analyst role (higher skill overlap).
  * Missing skills: SQL, Deep Learning.
  * Project Ideas:

    * SQL-based ETL Pipeline
    * Image Classifier with Deep Learning

---

## 🛠️ Tech Stack

* **Python** (pandas, numpy, scikit-learn)
* **NLP** (regex, spaCy for parsing)
* **Recommender Systems** (cosine similarity)
* **Visualization** (matplotlib, seaborn)
* **Streamlit** (for demo app)

---

## 📌 Next Steps

* Improve resume parser with ML-based NER.
* Add more curated project ideas.
* Deploy Streamlit app online (Heroku / Streamlit Cloud).

---

## 👨‍💻 Author

Built with ❤️ by *Arti Rani* for project submission (Sept 2025).
