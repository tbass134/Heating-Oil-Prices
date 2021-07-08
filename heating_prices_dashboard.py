from pandas.io.parsers import read_table
import streamlit
import boto3
from boto3.dynamodb.conditions import Key
from pycaret.regression import *
from fastai.tabular.all import *

from dotenv import load_dotenv
import os
load_dotenv() 

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)

dynamodb = session.resource('dynamodb')

import pandas as pd
import streamlit as st
import plotly.express as px

import datetime
table = dynamodb.Table('heating_oil_prices')


@st.cache()
def load_data():
    response = table.scan()
    df = pd.DataFrame(response["Items"])
    df["price150"] = df["price150"].astype(float)
    df["price300"] = df["price300"].astype(float)
    df["price500"] = df["price500"].astype(float)
    df.rename({
        "price150":"Price per 150 Gallons",
        "price300":"Price per 300 Gallons",
        "price500":"Price per 500 Gallons",
    }, inplace=True, axis=1)
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df = df.set_index("last_updated")
    df["state"] = df["state"].apply(lambda x: "NewYork" if str(x) == "nan" else x)
    df = df.sort_index()
    return df

def get_states():
   return df.state.value_counts().rename_axis('unique_values').reset_index(name='counts')["unique_values"]

def get_suppliers_by_state(state):
    return df[df["state"] == state]["supplier"].dropna()

def get_available_prices():
    return [col for col in df.columns if "Price" in col]

def pad_value(day):
    value = str(int(day))
    print()
    if len(value) ==1:
        return f'0{value}'
    return value

def get_forecast(end_date,model):
    forecast = pd.DataFrame({"preds":"","last_updated":pd.date_range(end_date, periods=8)})
    add_datepart(forecast, field_name="last_updated")
    forecast = forecast[['last_updatedYear', 'last_updatedMonth', 'last_updatedDay']]
    # loaded_model = load_model(model)
    my_model = load_model("/models/price150_model")
    preds = predict_model(my_model, data=forecast)["Label"]
    forecast["preds"] = preds

    dates = []
    for idx, x in forecast.iterrows():
        date_str =  f'{pad_value(x["last_updatedYear"])}-{pad_value(x["last_updatedMonth"])}-{pad_value(x["last_updatedDay"])}'
        dates.append(pd.to_datetime(date_str, format='%Y-%m-%d'))
    forecast["last_updated"] = dates
    forecast = forecast[['last_updated', 'preds']]
    forecast = forecast.set_index('last_updated')
    return forecast

def plot_data(supplier, price, state = "NewYork", start_date = "2020-12-01", end_date = "2021-04-16"):
    suppliers_by_state = df[ (df["state"] == state) & (df["supplier"] == supplier)][["supplier", price]].dropna()
    
    if start_date != None and end_date != None:
        mask = (suppliers_by_state.index >=start_date) &  (suppliers_by_state.index <=end_date)
        data = suppliers_by_state[mask]
    else:
        data = suppliers_by_state

    if len(data) < 10:
        return None
    else:
        fig = px.line(
            data, 
            y=price, 
            labels={
                "last_updated": "Date"
                },
            title = f'Price of oil from {supplier}')
        return fig

def plot_avg(df, state, show_forecast=True):
    avg_df = df.reset_index()
    avg_df = avg_df[ avg_df["state"] == state].resample('d', on='last_updated').mean().dropna()
    avg_df = df.reset_index()
    min_date = avg_df["last_updated"].min().strftime("%b %d, %Y")
    max_date = avg_df["last_updated"].max().strftime("%b %d, %Y")

    if show_forecast:
        forecast = get_forecast(max_date, "models/price150_model")
        print(forecast)
        pass

    fig = px.line(
                    avg_df, 
                    labels={
                                "last_updated": "Date",
                                "value": "Price ($)"
                            },
                    title = f'Average prices per day in  {selected_state} from {min_date} to {max_date}')
    return fig

df = load_data()
selected_state = st.sidebar.selectbox("Select State", get_states())
selected_suppliers = st.sidebar.selectbox("Supplier", get_suppliers_by_state(selected_state))
prices = st.sidebar.selectbox("Prices", get_available_prices())

# plot = plot_data(selected_suppliers, prices, selected_state, None, None)

# if plot != None:
#     st.plotly_chart(plot)
# else:
#     st.text("Not enough data to display")

st.plotly_chart(plot_avg(df, selected_state))
