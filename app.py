import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import dash_bootstrap_components as dbc


model = joblib.load("best_model1.pkl")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    html.H1("Bank Customer Churn Prediction App.", className="display-4"),
    html.Label("Credit Score"),
    dcc.Input(id="credit-score", type="number", min=0, max=1000, value=600, step=1, className="form-control"),
    html.Label("Age"),
    dcc.Input(id="age", type="number", min=18, max=100, value=30, step=1, className="form-control"),
    html.Label("Balance"),
    dcc.Input(id="balance", type="number", min=0.0, max=300000.0, value=10000.0, step=100.0, className="form-control"),
    html.Label("Number of Products"),
    dcc.Dropdown(id="num-products", options=[
        {'label': '1', 'value': 1},
        {'label': '2', 'value': 2},
        {'label': '3', 'value': 3},
        {'label': '4', 'value': 4}
    ], value=1, className="form-control"),
    html.Label("Estimated Salary"),
    dcc.Input(id="estimated-salary", type="number", min=0.0, max=300000.0, value=50000.0, step=100.0, className="form-control"),
    html.Button('Predict', id='predict-button', className="btn btn-primary"),
    html.Div(id='prediction-output', className="mt-4")
], className="mt-4")


@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        Input('credit-score', 'value'),
        Input('age', 'value'),
        Input('balance', 'value'),
        Input('num-products', 'value'),
        Input('estimated-salary', 'value')
    ]
)
def update_prediction(n_clicks, credit_score, age, balance, num_of_products, estimated_salary):
    if n_clicks is None:
        return ""
    
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Age": [age],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "EstimatedSalary": [estimated_salary]
    })

    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        return html.Div("The customer is not at risk of churn.", className="alert alert-success")
    else:
        return html.Div("The customer is at risk of churn.", className="alert alert-danger")


if __name__ == '__main__':
    app.run_server(debug=True)
