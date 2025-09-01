# House Price Prediction — Deployment Starter

This starter lets you serve your trained model via a FastAPI endpoint and/or a Streamlit app.

## Layout
```
house_price_deploy/
├── api/
│   └── main.py
├── model/
│   └── lasso_house_price_model.pkl
├── streamlit_app.py
├── requirements.txt
├── Dockerfile.api
└── Dockerfile.streamlit
```

## 1) Set up a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> Important: adjust `scikit-learn` and `numpy` versions in `requirements.txt` to match those used to train the model.

## 2) Run the API
```bash
uvicorn api.main:app --reload --port 8000
```
- Check: http://localhost:8000/health
- Inspect expected columns: http://localhost:8000/schema

Predict with curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"area":3000,"bedrooms":3,"bathrooms":2,"stories":2,"mainroad":"yes","guestroom":"no","basement":"no","hotwaterheating":"no","airconditioning":"yes","parking":1,"prefarea":"no","furnishingstatus":"unfurnished"}'
```

## 3) Run the Streamlit app
```bash
streamlit run streamlit_app.py
```
- In the sidebar: set the API base to your FastAPI URL (or check "Use local model" to load the pickle directly).

## 4) Docker
Build and run the API:
```bash
docker build -t house-api -f Dockerfile.api .
docker run --rm -p 8000:8000 house-api
```

Build and run the Streamlit app:
```bash
docker build -t house-ui -f Dockerfile.streamlit .
docker run --rm -p 8501:8501 house-ui
```

## Notes
- The API tries to infer training columns from the model. If it can't, it still accepts records and passes them to the model as-is.
- If predictions fail with a message about shapes or dtypes, ensure your input columns match the training names/types (see `/schema`).
- For cloud deploys: you can host the API on Render/Railway/Fly.io and the Streamlit app on Streamlit Cloud or Hugging Face Spaces.
