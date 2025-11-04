# **Pearls AQI Predictor**
*“Breathe smarter, see tomorrow’s air today.”*

An automated Machine learning pipeline that predicts Karachi’s Air Quality Index (AQI) for the next 3 days, built using Hopsworks Feature Store, Model Registry, and a Streamlit dashboard for live visualization.

---

## **Overview**

This project automates the full **AQI prediction lifecycle**:

1. **Fetches** live & historical weather + air quality data  
2. **Engineer** features and store them in **Hopsworks Feature Store**  
3. **Train** and **evaluate** ML models (**Random Forest** + **LSTM**)  
4. **Predict** AQI for the next 3 days  
5. **Visualize** forecasts and EDA insights with **Streamlit**  
6. **Automate** everything using **GitHub Actions (CI/CD)**  

---
## **Project Structure**

```
aqi_forecast/

├── .github/workflows/ci_cd_pipeline.yml

├── dashboard/
│ ├── dashboard.py
│ └── eda_outputs/

├── data/
│ ├── raw_aqicn/
│ ├── raw_openweather/
│ ├── raw_openmeteo/
│ ├── features/
│ └── predictions/

├── data_fetch/
│ ├── fetch_aqicn.py
│ ├── fetch_openweather.py
│ ├── fetch_meteostat.py

├── features//features/
│ ├── backfill.py
│ └── backfill_live.py

├── trainings/
│ ├── train_sklearn.py
│ ├── train_tf.py
│ └── predict.py
│
├── eda.py
├── requirements.txt
└── .env
```
---

## **Setup Instructions**

### **Clone Repository**

```bash
git clone https://github.com/<your-username>/AQI_Forecast.git
cd AQI_Forecast
```

### **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### **Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **Configure .env File**

Create a .env in the root directory:
```
OPENWEATHER_API_KEY=your_openweather_key
AQICN_TOKEN=your_aqicn_token
AQI_FORECAST_API_KEY=your_hopsworks_api_key
CITY=Karachi
LAT=24.8607
LON=67.0011

```

### **Authenticate Hopsworks**

```
python -m hopsworks.login
```
---

## **How It Works**

### **1. Data Collection**

Fetch data from APIs:

```
python data_fetch/fetch_aqicn.py
python data_fetch/fetch_openweather.py
python data_fetch/fetch_meteostat.py
```

### **2. Feature Engineering & Backfill**

Merge, clean, and upload to Hopsworks:

```
python features/features/backfill.py
python features/features/backfill_live.py
```

### **3. Train Models**

Train two different models:

```
python trainings/train_sklearn.py    # Random Forest
python trainings/train_tf.py         # LSTM
```

### **4. Generate Predictions**

```
python trainings/predict.py
```
Outputs → data/predictions/latest_predictions.csv

### **5. Perform EDA**

```
python eda.py
```
Generates visual insights → dashboard/eda_outputs/

### **6. Launch Dashboard**

```
streamlit run dashboard/dashboard.py
```
---

## **CI/CD Automation (GitHub Actions)**

Workflow: .github/workflows/ci_cd_pipeline.yml


| **Stage** | **Frequency** | **Description**              |
|--------------|------------------|--------------------------------|
| **Backfill** | Hourly           | Fetch new live data            |
| **Train**    | Daily / On Push  | Retrain and evaluate models    |
| **Predict**  | Hourly           | Generate and save AQI forecasts |
| **EDA**      | After Training   | Auto-refresh visualizations    |
| **Commit**   | Auto             | Push new outputs to GitHub     |


Trigger manually:
```
gh workflow run "AQI Forecast CI/CD Pipeline"
```
---

## **Models Used**


| Model             | Framework    | Description                       |
| ----------------- | ------------ | --------------------------------- |
| **Random Forest** | Scikit-learn | Baseline regression model         |
| **LSTM**          | TensorFlow   | Time-series AQI forecasting model |


**Metrics:** RMSE, MAE, R²

---

## **Key Outputs**
- `data/predictions/latest_predictions.csv` → 3-day AQI forecast  
- `dashboard/eda_outputs/` → Generated EDA visuals  
- `models/` → Saved model artifacts  
- `aqi_features` → Feature Store on Hopsworks

---

## **Dashboard Preview**

- **Today’s AQI Summary** — color-coded & mood-based  
- **Next 3-Day Forecast** — with interactive charts  
- **EDA Visuals** — trends, correlations, and feature importance (complete eda_outputs)

Run it via:
```
streamlit run dashboard/dashboard.py
```

---

## **Future Enhancements**

- Add **SHAP/LIME** explainability  
- Integrate **data validation** using *Great Expectations*  
- Extend to **multi-city forecasting**  
- Implement **real-time alerts** for hazardous AQI levels

---

## **Acknowledgments**

- **[AQICN](https://aqicn.org/api/)** — Air Quality API  
- **[OpenWeather](https://openweathermap.org/api)** — Weather API  
- **[Open-Meteo](https://open-meteo.com/)** — Historical Weather Data  
- **[Hopsworks](https://www.hopsworks.ai/)** — Feature Store & Model Registry  
- **[Streamlit](https://streamlit.io/)** — Dashboard Framework  

---

### **About the Project**
This project represents a **complete production grade MLOps pipeline**, automating:
- Data collection  
- Model training and evaluation  
- Forecast generation  
- Visualization and dashboarding  

All integrated seamlessly through **Hopsworks**, **GitHub Actions**, and **Streamlit** — enabling continuous, end-to-end AQI forecasting.
---

<p align="center">
  <b>Misbah Azhar</b>  
  <br>
  💻 Developer | Data & MLOps Enthusiast  
  <br>
  <a href="mailto:misbahazhar018@gmail.com">misbahazhar018@gmail.com</a>  
  <br>
  <i>Pearls AQI Predictor (2025)</i>
</p>

---

## **License**
**MIT License © 2025 Misbah Azhar**

---
































