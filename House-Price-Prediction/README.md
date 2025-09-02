# 🏠 House Price Prediction

A Machine Learning project that predicts **house prices** based on key features such as area, number of bedrooms, bathrooms, stories, parking, and various categorical features (main road, guest room, basement, etc.).

The project demonstrates a complete ML workflow — **data preprocessing, feature engineering, model selection, hyperparameter tuning, model evaluation, interpretability, and deployment** using a Streamlit web app.

---

## 📌 Features

* **Data Preprocessing**: Scaling numeric features & encoding categorical features using `ColumnTransformer`.
* **Feature Engineering**: Handling missing values, transforming features, and preparing raw data for modeling.
* **Modeling**: Trained and compared **Ridge Regression** and **Lasso Regression**.
* **Hyperparameter Tuning**: Used cross-validation to find the best alpha values.
* **Evaluation Metrics**: Evaluated models using **RMSE, MAE, and R²**.
* **Interpretability**: Compared Ridge vs Lasso coefficients to analyze feature importance.
* **Deployment**: Final pipeline (preprocessing + model) saved as `house_price_pipeline.pkl` and deployed via **Streamlit**.

---

## 🚀 Streamlit App

The app provides a simple interface where users can:

* Enter house details (area, bedrooms, bathrooms, etc.).
* Click **Predict** to get the estimated house price.

---

## 📂 Project Structure

```
House-Price-Prediction/
│── data/                 # Raw dataset (optional for repo)
│── notebooks/            # Jupyter notebooks for EDA, training, tuning
│── house_price_deploy/   # Deployment folder
│   ├── app.py            # Streamlit app
│   ├── house_price_pipeline.pkl  # Saved model
│── requirements.txt      # Project dependencies
│── README.md             # Project description
```

---

## ⚙️ Installation & Usage

1. Clone this repo:

   ```bash
   git clone https://github.com/raniartii/House-Price-Prediction.git
   cd House-Price-Prediction/house_price_deploy
   ```
2. Create a virtual environment & install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
4. Open your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 📊 Model Performance

* **Ridge Regression**

  * RMSE: \~1,147,142
  * MAE : \~891,609
  * R²  : \~0.67

* **Lasso Regression** (Final Winner 🏆)

  * RMSE: \~1,136,713
  * MAE : \~881,942
  * R²  : \~0.68

---

## 💡 Future Improvements

* Add more advanced models (Random Forest, XGBoost).
* Improve feature engineering with domain-specific insights.
* Deploy on cloud hosting (e.g., Hostinger, Heroku, Render).
* Add CI/CD pipeline for automated deployment.

---

## 👨‍💻 Author

* **Your Name** — Arti Rani
* Project built as part of learning & practice on **End-to-End Machine Learning pipelines**

---