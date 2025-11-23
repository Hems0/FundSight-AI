# app.py
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from datetime import datetime, timedelta

# Step 1: Simulate Startup Data with Enhanced Features
def generate_startup_data():
    dates = pd.date_range(start="2023-01-01", end="2025-01-31", freq="ME")
    num_startups = 200  # 25 * 200 = 5,000 records
    startups = [f"Startup_{i}" for i in range(1, num_startups + 1)]
    industries = ["Tech", "Health", "Finance", "AI"]
    data = []
    
    for date in dates:
        for startup in startups:
            funding = min(np.random.randint(100000, 5000000) * (1 + date.month / 12), 10000000)  # Cap at $10M
            employees = np.random.randint(10, 200) * (1 + date.month / 24)
            launches = np.random.randint(0, 3)
            industry = np.random.choice(industries)
            startup_age = (date - datetime(2023, 1, 1)).days / 365
            emp_growth = np.random.uniform(-0.1, 0.3)
            prev_funding = funding * np.random.uniform(0.5, 1.5) if date > dates[0] else 0
            emp_launch_interaction = employees * launches  # Interaction term
            data.append([date, startup, funding, employees, launches, industry, startup_age, emp_growth, prev_funding, emp_launch_interaction])
    
    df = pd.DataFrame(data, columns=["Date", "Startup", "Funding", "Employees", "Launches", "Industry", "Startup_Age", "Emp_Growth", "Prev_Funding", "Emp_Launch_Interaction"])
    print(f"Generated {len(df)} records")
    return df

# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")
    
    fig1 = px.histogram(df, x="Funding", nbins=50, title="Funding Distribution")
    st.plotly_chart(fig1)
    
    fig2 = px.scatter(df, x="Employees", y="Funding", color="Launches", 
                      hover_data=["Startup", "Date"], title="Funding vs Employees")
    st.plotly_chart(fig2)
    
    corr = df[["Funding", "Employees", "Launches", "Startup_Age", "Emp_Growth", "Prev_Funding", "Emp_Launch_Interaction"]].corr()
    fig3 = ff.create_annotated_heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        annotation_text=corr.round(2).values, showscale=True
    )
    fig3.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig3)
    
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    fig4 = px.box(df, x="Month", y="Funding", title="Funding Distribution by Month")
    st.plotly_chart(fig4)
    
    sample_startups = df["Startup"].unique()[:5]
    sample_df = df[df["Startup"].isin(sample_startups)]
    fig5 = px.line(sample_df, x="Date", y="Funding", color="Startup", 
                   title="Funding Trend for Sample Startups")
    st.plotly_chart(fig5)

# Step 3: XGBoost Model with Enhanced Tuning
def train_xgb_model(df):
    # Preprocessing
    df_encoded = pd.get_dummies(df, columns=["Industry"], drop_first=True)
    X = df_encoded[["Employees", "Launches", "Startup_Age", "Emp_Growth", "Prev_Funding", "Emp_Launch_Interaction",
                    "Industry_Finance", "Industry_Health", "Industry_Tech"]]
    y = df_encoded["Funding"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }
    xgb_model = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=20, cv=5, scoring="r2", n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    
    # Best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Regression Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Classification Metrics (threshold: median funding)
    threshold = np.median(y)
    y_test_binary = (y_test > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    
    # Display Metrics
    st.subheader("XGBoost Model Performance")
    st.write(f"R² Score: {r2:.3f}")
    st.write(f"MAE: ${mae:,.0f}")
    st.write(f"RMSE: ${rmse:,.0f}")
    st.write(f"MAPE: {mape:.2f}%")
    st.write(f"Accuracy (High vs Low Funding): {accuracy:.3f}")
    st.write(f"Precision: {precision:.3f}")
    st.write(f"Recall: {recall:.3f}")
    st.write(f"F1-Score: {f1:.3f}")
    
    # Predict on full dataset
    df_encoded["Predicted_Funding"] = best_model.predict(X_scaled)
    return df_encoded, best_model, scaler

# Step 4: Time-Series Forecast with Regressors
def train_time_series_model(df):
    ts_df = df.groupby("Date").agg({"Funding": "sum", "Employees": "mean", "Launches": "sum", "Emp_Growth": "mean"}).reset_index()
    ts_df.columns = ["ds", "y", "Employees", "Launches", "Emp_Growth"]
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.add_regressor("Employees")
    model.add_regressor("Launches")
    model.add_regressor("Emp_Growth")
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=6, freq="ME")
    future["Employees"] = ts_df["Employees"].iloc[-1]
    future["Launches"] = ts_df["Launches"].iloc[-1]
    future["Emp_Growth"] = ts_df["Emp_Growth"].iloc[-1]
    forecast = model.predict(future)
    return forecast, model

# Step 5: Dashboard
def create_dashboard(df, forecast):
    st.title("Startup Ecosystem Tracker (5,000 Records)")
    
    # EDA Section
    perform_eda(df)
    
    # Filter by Startup
    startup_filter = st.selectbox("Select Startup", options=["All"] + df["Startup"].unique().tolist(), index=0)
    if startup_filter != "All":
        filtered_df = df[df["Startup"] == startup_filter]
    else:
        filtered_df = df
    
    # Funding Trend
    st.subheader("Funding Trends")
    fig6 = px.line(filtered_df.groupby("Date")["Funding"].sum(), title="Total Funding Over Time")
    st.plotly_chart(fig6)
    
    # XGBoost Predictions
    st.subheader("Predicted vs Actual Funding (XGBoost)")
    fig7 = px.scatter(filtered_df, x="Employees", y="Funding", color="Launches", 
                      hover_data=["Startup", "Date"], title="Actual Funding")
    fig7.add_scatter(x=filtered_df["Employees"], y=filtered_df["Predicted_Funding"], 
                     mode="markers", name="Predicted", marker=dict(color="red"))
    st.plotly_chart(fig7)
    
    # Time-Series Forecast
    st.subheader("Funding Forecast")
    fig8 = px.line(forecast, x="ds", y="yhat", title="Funding Forecast")
    fig8.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound")
    fig8.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound")
    st.plotly_chart(fig8)
    
    # Key Metrics
    st.subheader("Key Metrics")
    total_funding = filtered_df["Funding"].sum() / 1e6
    avg_employees = filtered_df["Employees"].mean()
    total_launches = filtered_df["Launches"].sum()
    st.write(f"Total Funding: ${total_funding:.2f}M")
    st.write(f"Average Employees: {avg_employees:.1f}")
    st.write(f"Total Product Launches: {total_launches}")

# Main Execution
if __name__ == "__main__":
    # Generate data
    df = generate_startup_data()
    
    # Train models
    df, xgb_model, scaler = train_xgb_model(df)
    forecast, ts_model = train_time_series_model(df)
    
    # Launch dashboard
    create_dashboard(df, forecast)