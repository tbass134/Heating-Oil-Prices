import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd
from fastai.tabular.all import *
from pycaret.regression import *


dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('heating_oil_prices')

def get_data():
    response = table.scan()
    df = pd.DataFrame(response["Items"])
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df["state"] = df["state"].apply(lambda x: "NewYork" if str(x) == "nan" else x)
    df = df.sort_index()
    return df
    
def train(data, price):

    mask = round(len(data) * 0.7)
    train = data[:mask]
    test = data[mask:]
    print("train",train.shape)
    print("test",test.shape)

    setup(data = train, test_data = test, target = price, fold_strategy = 'timeseries', numeric_features = ['last_updatedYear','last_updatedMonth', 'last_updatedDay'], fold=2, transform_target = True, session_id = 42)
    print("running compare models")
    best = compare_models(sort = 'MAE')
    print("best model", best)

    model_name = f'{price}_model'
    save_model(best, model_name)
    print(f'model saved to {model_name}')

if __name__ == "__main__":
    df = get_data()
    df = df[ df["state"] == "NewYork"].resample('d', on='last_updated').mean().dropna()
    df = df_equal.reset_index()
    add_datepart(df, field_name="last_updated")
    df = df[['price500', 'price300', 'price150', 'last_updatedYear', 'last_updatedMonth', 'last_updatedDay']]

    for feature in ["price150", "price300", "price500"]:
        train(df, feature)
