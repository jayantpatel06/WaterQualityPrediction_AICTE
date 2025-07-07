# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

# Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Safety Thresholds
SAFETY_LIMITS = {
    'human': {
        'NH4': 0.5, 'NO3': 50, 'NO2': 0.1, 
        'SO4': 250, 'CL': 250
    },
    'fish': {
        'NH4': 0.5, 'BSK5': 3, 'Suspended': 25, 
        'O2': 5, 'NO2': 0.1, 'PO4': 0.1, 'CL': 230
    }
}

def check_safety(pollutant, value):
    human_safe = value <= SAFETY_LIMITS['human'].get(pollutant, float('inf'))
    fish_safe = (value >= SAFETY_LIMITS['fish'].get('O2', -np.inf) if pollutant == 'O2' 
                else value <= SAFETY_LIMITS['fish'].get(pollutant, float('inf')))
    return human_safe, fish_safe

# Streamlit UI
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        # Prepare input
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])
        
        # Align columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        # Create results table
        results = []
        for p, val in zip(pollutants, predicted_pollutants):
            human_safe, fish_safe = check_safety(p, val)
            
            human_limit = SAFETY_LIMITS['human'].get(p, 'N/A')
            fish_limit = SAFETY_LIMITS['fish'].get(p, 'N/A')
            
            # For O2, fish need > limit (special case)
            if p == 'O2':
                fish_limit = f"> {SAFETY_LIMITS['fish'].get(p, 'N/A')}"
            
            results.append({
                'Pollutant': p,
                'Predicted (mg/L)': f"{val:.2f}",
                'Human Limit (mg/L)': human_limit,
                'Human Safe': '✅' if human_safe else '❌',
                'Fish Limit (mg/L)': fish_limit,
                'Fish Safe': '✅' if fish_safe else '❌'
            })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Display results
        st.subheader(f"Predicted pollutant levels for station '{station_id}' in {year_input}:")
        
        # Style the table
        st.dataframe(results_df,hide_index=True,use_container_width=True)
    
        # Summary alerts
        human_unsafe = results_df[results_df['Human Safe'] == '❌']['Pollutant'].tolist()
        fish_unsafe = results_df[results_df['Fish Safe'] == '❌']['Pollutant'].tolist()
        
        if human_unsafe:
            st.error(f"⚠️ Unsafe for humans: {', '.join(human_unsafe)} exceed limits")
        if fish_unsafe:
            st.error(f"🐟 Unsafe for fish: {', '.join(fish_unsafe)} exceed limits")
        
        if not human_unsafe and not fish_unsafe:
            st.success("✅ Water is safe for both humans and fish!")