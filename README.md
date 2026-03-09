# 💼 Women Salary Prediction — INE Fecundity Survey 2018 (Spain)

A machine learning regression project that predicts women's salaries based on the **Spanish National Statistics Institute (INE) Fecundity Survey 2018**. The project includes a full ML pipeline — from raw data ingestion to model serving via a FastAPI backend and a Streamlit frontend.

---

## 🗂️ Project Structure

```
├── src/
│   ├── feature_pipeline/       # Data loading, preprocessing & feature engineering
│   ├── training_pipeline/      # Model training & evaluation
│   └── inference_pipeline/     # Prediction serving
├── Dockerfile                  # API image
├── Dockerfile.streamlit        # Streamlit image
├── pyproject.toml
└── .github/
    └── workflows/
        └── ci.yml              # CI/CD pipeline (Docker Hub)
```

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/mauronperez/salary-reg.git
cd salary-regression-mle

# Install dependencies
pip install -e .
```

---

## 🚀 Running the Pipeline

### 1. Load and split raw data
```bash
python src/feature_pipeline/load.py
```

### 2. Preprocess splits
```bash
python -m src.feature_pipeline.preprocess
```

### 3. Feature engineering
```bash
python -m src.feature_pipeline.feature_engineering
```

### 4. Train the model
```bash
python -m src.training_pipeline.train
```

### 5. Run inference
```bash
python -m src.inference_pipeline.predict
```

---

## 🐳 Running with Docker

```bash
# Pull images from Docker Hub
docker pull mauronp/salary-api:latest
docker pull mauronp/salary-streamlit:latest

# Run API
docker run -p 8000:8000 mauronp/salary-api:latest

# Run Streamlit
docker run -p 8501:8501 mauronp/salary-streamlit:latest
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

---

## 📦 Data Pipeline

```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py

# 2. Preprocess splits
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
```

---

## 🏋️ Training Pipeline

```bash
# Train baseline model
python src/training_pipeline/train.py

# Hyperparameter tuning with MLflow
python src/training_pipeline/tune.py

# Model evaluation
python src/training_pipeline/eval.py
```

---

## 🔮 Inference

```bash
# Single inference
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv

```

---

## 🌐 API Service

```bash
# Start FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 📊 Streamlit Dashboard

```bash
# Start Streamlit dashboard locally
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🐳 Running with Docker

```bash
# Build API container
docker build -t salary-regression .

# Build Streamlit container
docker build -t salary-streamlit -f Dockerfile.streamlit .

# Run API container
docker run -p 8000:8000 salary-regression

# Run Streamlit container
docker run -p 8501:8501 salary-streamlit
```

Or with Docker Compose:
```bash
docker-compose up
```

---

## 📈 MLflow Tracking

```bash
# Start MLflow UI (view experiments)
mlflow ui
```

---

## 🔄 CI/CD

On every push to `main`, GitHub Actions automatically:

1. Builds the `salary-api` Docker image → pushes to `mauronp/salary-api:latest`
2. Builds the `salary-streamlit` Docker image → pushes to `mauronp/salary-streamlit:latest`

---

## 📊 Data Source

- **Survey:** [Encuesta de Fecundidad 2018 — INE](https://www.ine.es/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736177006&menu=resultados&idp=1254735573002)
- **Target variable:** Women's salary
- **Features:** Age, education level, number of children, employment type, region, and more

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Pandas, Polars, NumPy |
| Feature engineering | Scikit-learn, Category Encoders |
| Modeling | LightGBM, XGBoost, Scikit-learn |
| Experiment tracking | MLflow, Optuna |
| Data validation | Great Expectations, Evidently |
| API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions + Docker Hub |

---