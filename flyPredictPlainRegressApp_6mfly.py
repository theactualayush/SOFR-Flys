import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import requests
import datetime
import calendar

# Date Math Functions
def get_third_wednesday(year, month):
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    wednesdays = [day for week in monthcal for day in week if day.weekday() == calendar.WEDNESDAY and day.month == month]
    return wednesdays[2]

def generate_active_contracts(start_date=None, num_contracts=18):
    if start_date is None:
        start_date = datetime.date.today()
        
    contract_codes = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
    
    test_dates = [
        (start_date.year - 1, 12),
        (start_date.year, 3),
        (start_date.year, 6),
        (start_date.year, 9),
        (start_date.year, 12)
    ]
    
    front_year = None
    front_month = None
    
    for i in range(len(test_dates) - 1):
        prev_y, prev_m = test_dates[i]
        next_y, next_m = test_dates[i+1]
        
        roll_date = get_third_wednesday(next_y, next_m)
        
        if start_date <= roll_date:
            front_year = prev_y
            front_month = prev_m
            break
            
    if front_year is None:
        front_year = start_date.year
        front_month = 12
        
    contracts = []
    curr_month = front_month
    curr_year = front_year
    
    for _ in range(num_contracts):
        code = contract_codes[curr_month]
        yr_str = str(curr_year)[-2:]
        contracts.append(f"SRA{code}{yr_str}")
        
        curr_month += 3
        if curr_month > 12:
            curr_month = 3
            curr_year += 1
            
    return contracts

# Fetch Data from API 
@st.cache_data(show_spinner="Fetching data from API...")
def fetch_and_calculate_flies(token, instruments, count=200):
    url = "https://qh-api.corp.hertshtengroup.com/api/v2/ohlc/"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "instruments": ",".join(instruments),
        "interval": "1D",
        "count": count
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
        return None
        
    data = response.json()
    if not data:
        st.warning("No data returned from API.")
        return None
        
    # Convert JSON to DataFrame
    df_raw = pd.DataFrame(data)
    
    # Convert 'time' to datetime and sort
    df_raw['Date'] = pd.to_datetime(df_raw['time'], unit='ms')
    
    # Pivot so columns are the products (e.g. SRAH24) and rows are dates, values are 'close'
    df_pivot = df_raw.pivot(index='Date', columns='product', values='close')
    df_pivot = df_pivot.sort_index()

    # Create 6-Month Flies
    # Legs are spaced 2 quarters apart: (i, i+2, i+4)
    # e.g. M26-Z26-M27, U26-H27-U27
    available_legs = [inst for inst in instruments if inst in df_pivot.columns]
    
    fly_df = pd.DataFrame(index=df_pivot.index)
    
    # We need at least 5 contracts to make one 6-month fly (indices 0, 2, 4)
    for i in range(len(available_legs) - 4):
        leg1 = available_legs[i]
        leg2 = available_legs[i + 2]  # 6 months forward
        leg3 = available_legs[i + 4]  # 12 months forward
        
        fly_name = f"{leg1}-{leg2}-{leg3}"
        # Fly = Leg1 - 2*Leg2 + Leg3
        fly_df[fly_name] = (df_pivot[leg1] - 2 * df_pivot[leg2] + df_pivot[leg3]) * 100
        
    return fly_df.dropna(how='all')

# Regression Function
def run_regression(df, reference_fly, window=55):
    # Determine which columns are flies. First column is assumed to be Date, so we skip it.
    fly_columns = df.columns.dropna()[1:]

    if reference_fly not in fly_columns:
        st.error("Selected fly not found in data.")
        return None

    ref_idx = fly_columns.get_loc(reference_fly)
    relevant_flies = fly_columns[max(0, ref_idx - 1):]

    df_filtered = df[[df.columns[0]] + list(relevant_flies)].dropna()
    
    if len(df_filtered) > window:
        df_filtered = df_filtered.tail(window)
        st.info(f"Using the last {window} valid days for rolling regression.")
    else:
        st.warning(f"Only {len(df_filtered)} valid days available, which is less than the {window}-day window.")
        
    X = df_filtered[[reference_fly]].values

    results = []
    for fly in relevant_flies:
        if fly == reference_fly:
            continue
        y = df_filtered[fly].values
        model = LinearRegression().fit(X, y)

        y_pred = model.predict(X)
        n = len(y)
        p = X.shape[1] 
        residuals = y - y_pred
        mse_resid = np.sum(residuals ** 2) / (n - p - 1) if n > p + 1 else np.nan
        std_error = np.sqrt(mse_resid) if not np.isnan(mse_resid) else np.nan

        results.append({
            "Target Fly": fly,
            "Coef": model.coef_[0],
            "Intercept": model.intercept_,
            "R2": model.score(X, y),
            "Std Error": std_error
        })
    return pd.DataFrame(results)

# Streamlit UI
st.sidebar.header("API & Settings")
# Hardcoded API Token
api_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoyMDg3NDYyODc4LCJpYXQiOjE3NzIxMDI4NzgsImp0aSI6IjcyMGU4NmEyOGE1ZTQ4ZDNhODdhYmEyM2IwZGJlNDhjIiwidXNlcl9pZCI6MzY5fQ.bUs-KmTnTh8Wcf-1X0UUV0LnYUs_R8YQmKAcM0LYw-E"

num_contracts = st.sidebar.number_input("Number of Contracts to Fetch", min_value=5, max_value=40, value=18, step=1)

# Dynamically generate contracts based on today's date
instruments_list = generate_active_contracts(num_contracts=num_contracts)
st.sidebar.write("**Target Contracts:**")
st.sidebar.text(", ".join(instruments_list))

rolling_window = st.sidebar.number_input("Rolling Window (Days)", min_value=5, max_value=1000, value=55, step=1)

st.title("SOFR 6-Month Fly Curve Regression App")

if api_token and instruments_list:
    
    if st.sidebar.button("Fetch Data"):
        # Clear cache inside a button click will force reload if needed, 
        # but otherwise it relies on the function state.
        pass
        
    df = fetch_and_calculate_flies(api_token, instruments_list, count=200)
    
    if df is not None and not df.empty:
        # fly_options uses the calculated columns
        fly_options = df.columns.tolist()
        
        if len(fly_options) == 0:
            st.warning("Not enough instruments to calculate a 6-month Fly (need at least 5 consecutive contracts).")
        else:
            selected_fly = st.selectbox("Select the base fly:", fly_options)

            if selected_fly:
                # To align with original run_regression skipping "first column as date",
                # let's reset index so Date is the first column.
                df_reset = df.reset_index()
                
                regression_df = run_regression(df_reset, selected_fly, window=rolling_window)
                
                if regression_df is not None:
                    st.subheader(f"Regression Results using {selected_fly} as base fly")
                    st.dataframe(regression_df)

                    current_price = st.number_input(f"Enter live price for {selected_fly}", value=0.0, step=0.01)

                    if st.button("Predict Curve"):
                        regression_df["Predicted Price"] = regression_df["Coef"] * current_price + regression_df["Intercept"]
                        
                        # Get the latest actual prices from the dataframe to act as "Current"
                        latest_data = df_reset.iloc[-1]
                        live_date = pd.to_datetime(latest_data['Date']).strftime('%Y-%m-%d')
                        
                        # Build comparison dataframe
                        comparison_data = []
                        # Base fly comparison (Actual and Predicted are both the user-entered price for the base fly)
                        comparison_data.append({
                            "Target Fly": selected_fly,
                            "Live Price": current_price,
                            "Predicted Price": current_price
                        })
                        
                        for idx, row in regression_df.iterrows():
                            fly_name = row["Target Fly"]
                            # Safely attempt to get the most recent data point for the target fly
                            live_val = latest_data.get(fly_name, np.nan)
                            
                            comparison_data.append({
                                "Target Fly": fly_name,
                                "Live Price": live_val,
                                "Predicted Price": row["Predicted Price"]
                            })
                            
                        comparison_df = pd.DataFrame(comparison_data)

                        st.subheader("Live vs Predicted Prices")
                        st.caption(f"Live prices as of: **{live_date}**")
                        
                        # Display both as a Table
                        # Format the DataFrame to look cleaner in Streamlit
                        st.dataframe(comparison_df.style.format({
                            "Live Price": "{:.4f}",
                            "Predicted Price": "{:.4f}"
                        }))

                        # Display as a Line Chart
                        st.subheader("Curve Comparison Chart")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Plot Live Prices
                        ax.plot(comparison_df["Target Fly"], comparison_df["Live Price"], marker='o', label='Live Price', color='blue')
                        
                        # Plot Predicted Prices
                        ax.plot(comparison_df["Target Fly"], comparison_df["Predicted Price"], marker='x', linestyle='--', label='Predicted Price', color='orange')
                        
                        ax.set_title(f"Live vs Predicted 6M Fly Curve (Base: {selected_fly})")
                        ax.set_ylabel("Price")
                        ax.tick_params(axis='x', rotation=45)
                        ax.legend()
                        ax.grid(True, linestyle=':', alpha=0.6)
                        
                        # Improve layout
                        plt.tight_layout()
                        st.pyplot(fig)
else:
    st.info("Please enter your API Bearer Token and Instrument list in the sidebar to fetch data.")
