# Water Quality Prediction - RMS

This project aims to predict multiple water quality parameters using machine learning techniques, specifically `MultiOutputRegressor` wrapped around a `RandomForestRegressor`. It is developed as part of a one-month **AICTE Virtual Internship sponsored by Shell** in **June 2025**.

---

## Overview

Access to clean water is a critical global concern. This solution provides:

- Real-time prediction of 9 key water quality parameters
- Safety assessment against human and aquatic life thresholds
- Early pollution detection for timely intervention

Key project components:
- Data collection & preprocessing of real-world water quality datasets
- Multi-target regression using supervised ML
- Model pipeline with `MultiOutputRegressor(RandomForestRegressor())`
- Comprehensive model evaluation
- Deployment via Streamlit web interface

---

## Features

- **Multi-parameter prediction**: Simultaneously estimates levels of 9 pollutants
- **Safety evaluation**: Compares results against human/fish safety thresholds
- **User-friendly interface**: Simple inputs (year & station ID) generate detailed reports
- **Visual alerts**: Clear indicators for unsafe conditions
---

## Technologies Used

- **Python 3.12** - Core language
- **Pandas, NumPy** – Data handling and preprocessing
- **Scikit-learn** – Machine learning implementation
- **Matplotlib, Seaborn** – Data visualization
- **Joblib** – Model persistence
- **Streamlit** – Web application deployment
- **Jupyter Notebook** – Interactive development

---

## Predicted Water Quality Parameters

The model predicts multiple water quality parameters such as:

- NH4
- BOD5 (BSK5)
- Suspended (Colloids)
- O2, NO3, NO2, SO4, PO4 and CL
---

## Model Performance

The model was evaluated using:

- **R² Score**
- **Mean Squared Error (MSE)**

Performance was acceptable across all parameters

---
## Internship Details

- **Internship Type**: AICTE Virtual Internship - Edunet Foundation
- **Sponsor**: Shell  
- **Duration**: June 2025 (1 month)  
- **Focus Area**: Machine Learning in Environmental Monitoring  

---
