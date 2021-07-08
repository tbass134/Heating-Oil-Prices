# Predicting the price of Heating Oil using PyCaret
In this notebook, we'll go over how to perform a time series forecasting on the price of heating oil.

## Background
In update New York, we are unable to get natural gas to service our home for heating. This is because the rock is mostly made of shale, which makes it tough to get pull natural gas. So we have to rely on oil for heat.

As the price of gas goes up and down, the price of heating oil is the same. There are many companies that distribute oil and all of them have different prices. Also, these prices change between seasons. So, I wanted to build an application that predicts the price of oil, so that I know when is the best time to buy.

## Data
Before building the models, i needed to get data. I have been unable to find a dataset with the current price of oil, therefore I had to build my own. The website cheapestoil.com shows the price of oil for many companies in the northeast United States. The site shows the latest prices for these companies, but they do not show the previous prices. 

So in order to get the prices, I build a AWS lambda function that scrapes the price of oil daily. I used AWS CloudWatch events to run a lambda function every 12 hours, in order to fetch the prices for that time. This lambda extracted the last updated date, price and supplier, and save these results as JSON and save  to an S3 bucket. After the json data is saved, I have another lambda function, attached as a trigger, to read each json file, and save into DynamoDB.  Please see this [GitHub repo](https://github.com/tbass134/Heating-Oil-Prices) for more detail on how I build these lambda functions. 

I started this project back in Dec 2020 in order to build up my dataset. The lambda function has been running for about 6 months, and I have a decent amount of data to work with. In order to expand my dataset, I was able to pull more data using The Wayback Machine on web.archive.org. The Wayback Machine stores a snapshot of many pages on the internet. It doesn't have every site, but it did have some snapshots from cheaptestoil.com. To get that data, I used https://github.com/hartator/wayback-machine-downloader to download the archive data. The archive only had 7 snapshots, between the dates of Aug 2020 and Oct 2021. 

In all, I have about 5k records of all the oil prices from the website

## Fetching Saved Data
I used DynamoDB to store the oil price data, and used Boto3 to fetch the data, which I then save to a CSV.



```python
import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd
dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table('heating_oil_prices')
response = table.scan()
df = pd.DataFrame(response["Items"])
df.to_csv("data.csv")
```


```python
def get_data():
    df = pd.read_csv("data.csv", usecols=["last_updated", "price150","price500", "price300", "supplier", "state"])
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df = df.set_index("last_updated")
    df["state"] = df["state"].apply(lambda x: "NewYork" if str(x) == "nan" else x)
    df = df.sort_index()
    return df
df = get_data()
```

On Cheapestoil.com, the have the price of oil in gallons, but the price is slightly different for how many gallons you buy. If we get suppliers in the state of New York, we'll see the following


```python
    state = "NewYork"
    suppliers_by_state = df[ (df["state"] == state)].dropna()
    suppliers_by_state.iloc[1]

```




    price500          1.449
    price300          1.469
    price150          1.549
    supplier    Suffolk Oil
    state           NewYork
    Name: 2020-08-03 15:11:16, dtype: object



The row above shows that the price for 500 gallons(`price500`) is $1.449 per gallon, 300 gallons(`price300`) is $1.69 and 150 gallons(`price150`) is $1.549

Lets see how many suppliers we have for New York


```python
suppliers_by_state["supplier"].value_counts().sum()
```




    2708



Since we have so many suppliers, a forecasr for the average price of oil for all the suppliers might be a better way to go, since the prices are simlar between every company


```python
df = df.reset_index()
data = df[ df["state"] == state].resample('d', on='last_updated').mean().dropna()
data = data.reset_index()
```

Here is the mean price of oil over all the suppliers in New York.


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_updated</th>
      <th>price500</th>
      <th>price300</th>
      <th>price150</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-08-03</td>
      <td>1.672000</td>
      <td>1.719400</td>
      <td>1.76140</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-08-04</td>
      <td>1.632429</td>
      <td>1.625750</td>
      <td>1.68075</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-29</td>
      <td>1.852500</td>
      <td>1.819455</td>
      <td>1.85300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-11-11</td>
      <td>1.786250</td>
      <td>1.759000</td>
      <td>1.76900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-11-12</td>
      <td>1.420000</td>
      <td>1.460000</td>
      <td>1.54000</td>
    </tr>
  </tbody>
</table>
</div>



For a quick check on our data, let's plot the price for 500 gallons


```python
data["price500"].plot()

```




    <AxesSubplot:>




    
![svg](HeatingOilPrices-TimeSeries_files/HeatingOilPrices-TimeSeries_13_1.svg)
    


# Model

Now, we can start building our model. We'll be using PyCaret to build our time series forecast. 
Before modeling, we need to update the dataset to remove the date and replace with numeric values. To do this, I've included fastai's `add_datepart` function to convert the data is series of features, split by year, month, day, day of week, and much more


```python
from fastai.tabular.all import *
add_datepart(data, field_name="last_updated")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price500</th>
      <th>price300</th>
      <th>price150</th>
      <th>last_updatedYear</th>
      <th>last_updatedMonth</th>
      <th>last_updatedWeek</th>
      <th>last_updatedDay</th>
      <th>last_updatedDayofweek</th>
      <th>last_updatedDayofyear</th>
      <th>last_updatedIs_month_end</th>
      <th>last_updatedIs_month_start</th>
      <th>last_updatedIs_quarter_end</th>
      <th>last_updatedIs_quarter_start</th>
      <th>last_updatedIs_year_end</th>
      <th>last_updatedIs_year_start</th>
      <th>last_updatedElapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.672000</td>
      <td>1.719400</td>
      <td>1.761400</td>
      <td>2020</td>
      <td>8</td>
      <td>32</td>
      <td>3</td>
      <td>0</td>
      <td>216</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.596413e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.632429</td>
      <td>1.625750</td>
      <td>1.680750</td>
      <td>2020</td>
      <td>8</td>
      <td>32</td>
      <td>4</td>
      <td>1</td>
      <td>217</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.596499e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.852500</td>
      <td>1.819455</td>
      <td>1.853000</td>
      <td>2020</td>
      <td>10</td>
      <td>44</td>
      <td>29</td>
      <td>3</td>
      <td>303</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.603930e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.786250</td>
      <td>1.759000</td>
      <td>1.769000</td>
      <td>2020</td>
      <td>11</td>
      <td>46</td>
      <td>11</td>
      <td>2</td>
      <td>316</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.605053e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.420000</td>
      <td>1.460000</td>
      <td>1.540000</td>
      <td>2020</td>
      <td>11</td>
      <td>46</td>
      <td>12</td>
      <td>3</td>
      <td>317</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.605139e+09</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2.220000</td>
      <td>2.260000</td>
      <td>2.340000</td>
      <td>2021</td>
      <td>6</td>
      <td>26</td>
      <td>28</td>
      <td>0</td>
      <td>179</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.624838e+09</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2.632621</td>
      <td>2.626935</td>
      <td>2.646290</td>
      <td>2021</td>
      <td>7</td>
      <td>26</td>
      <td>2</td>
      <td>4</td>
      <td>183</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.625184e+09</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2.606429</td>
      <td>2.582633</td>
      <td>2.611437</td>
      <td>2021</td>
      <td>7</td>
      <td>27</td>
      <td>5</td>
      <td>0</td>
      <td>186</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.625443e+09</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2.619422</td>
      <td>2.609042</td>
      <td>2.632478</td>
      <td>2021</td>
      <td>7</td>
      <td>27</td>
      <td>6</td>
      <td>1</td>
      <td>187</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.625530e+09</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2.571294</td>
      <td>2.545455</td>
      <td>2.575238</td>
      <td>2021</td>
      <td>7</td>
      <td>27</td>
      <td>7</td>
      <td>2</td>
      <td>188</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.625616e+09</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 16 columns</p>
</div>



The `add_datepart` function generates lots of feature for the date, but we don't need to use all of them. For this model, we'll use `last_updatedYear`,  `last_updatedMonth` and `last_updatedDay`. In future models, we can try to use other features.


```python
data = data[['price500', 'last_updatedYear', 'last_updatedMonth', 'last_updatedDay']]
```


```python
mask = round(len(data) * 0.7)
train = data[:mask]
test = data[mask:]
train.shape, test.shape
```




    ((43, 4), (18, 4))



After we split the data into a train, test set, we then must initalize the regression setup in PyCaret. This includes suppling the dataset, the feature we are predicting on (`price500`) and the other features to use(`last_updatedYear`,`last_updatedMonth` and `last_updatedDay`)


```python
from pycaret.regression import *
s = setup(data = train, test_data = test, target = 'price500', fold_strategy = 'timeseries', numeric_features = ['last_updatedYear','last_updatedMonth', 'last_updatedDay'], fold = 3, transform_target = True, session_id = 42)
```


Next we call the `compare_models` function to find the best model using the Mean Absolute Error(MAE), which is the mean of the absolute difference between the  models prediction and expected values.   


```python
best = compare_models(sort = 'MAE')
```

From the results above, it looks like the	Passive Aggressive Regressor has the lowest MAE error(0.0605), so we'll use that model on the test set


```python
preds = predict_model(best)
```

Next, we'll use this model and generate a forecast for the next 7 days worth of prices 



```python
forecast_df = pd.DataFrame()
forecast_df["last_updated"] = pd.date_range(start='2021-07-08', periods=8)
add_datepart(forecast_df, 'last_updated')
forecast_df = forecast_df[['last_updatedYear', 'last_updatedMonth', 'last_updatedDay']]
forecast_df
```


```python
predictions = predict_model(best, data=forecast_df)
predictions
```

Once we make our predictions, we'll then merge the predictions to the orginal data and plot the last 15 days in the dataframe


```python
def pad_value(day):
    value = str(int(day))
    print()
    if len(value) ==1:
        return f'0{value}'
    return value

results_df = pd.concat([data,predictions], axis=0)
dates = []
for idx, x in results_df.iterrows():
    date_str =  f'{pad_value(x["last_updatedYear"])}-{pad_value(x["last_updatedMonth"])}-{pad_value(x["last_updatedDay"])}'
    dates.append(pd.to_datetime(date_str, format='%Y-%m-%d'))
results_df["date"] = dates

results_df.drop(["last_updatedYear", "last_updatedMonth", "last_updatedDay"], axis=1,inplace=True)
results_df = results_df.set_index('date')

```


```python
results_df[-15:].plot()
```

The blue line above shows the actual prices and the orange are the predicitons, which do not look the best. If we zoom in to the predictions. we'll see a increasing in price per day


```python
results_df[-8:].plot()
```

# Saving The Model

After we trained our model, we can now save and use for forecasting on other data


```python
save_model(best, "model")
```

    Transformation Pipeline and Model Succesfully Saved





    (Pipeline(memory=None,
              steps=[('dtypes',
                      DataTypes_Auto_infer(categorical_features=[],
                                           display_types=True, features_todrop=[],
                                           id_columns=[], ml_usecase='regression',
                                           numerical_features=['last_updatedYear',
                                                               'last_updatedMonth',
                                                               'last_updatedDay'],
                                           target='price500', time_features=[])),
                     ('imputer',
                      Simple_Imputer(categorical_strategy='not_available',
                                     fill_value_ca...
                                                      regressor=PassiveAggressiveRegressor(C=1.0,
                                                                                           average=False,
                                                                                           early_stopping=False,
                                                                                           epsilon=0.1,
                                                                                           fit_intercept=True,
                                                                                           loss='epsilon_insensitive',
                                                                                           max_iter=1000,
                                                                                           n_iter_no_change=5,
                                                                                           random_state=42,
                                                                                           shuffle=True,
                                                                                           tol=0.001,
                                                                                           validation_fraction=0.1,
                                                                                           verbose=0,
                                                                                           warm_start=False),
                                                      shuffle=True, tol=0.001,
                                                      validation_fraction=0.1,
                                                      verbose=0,
                                                      warm_start=False)]],
              verbose=False),
     'model.pkl')



# Training Model for all Prices

After we have a intial model, we can now train 3 seperate models for each price(`price150`, `price300` and `price500`)
We'll refactor the model training code into a function that trains each price, gets the model with the best score, and saves the model to a seperate file


```python
def train(data, price, state="NewYork"):
    data = data.reset_index()
    data = data[ data["state"] == state].resample('d', on='last_updated').mean().dropna()
    data = data.reset_index()
   
    add_datepart(data, field_name="last_updated")
    data = data[[price, 'last_updatedYear', 'last_updatedMonth', 'last_updatedDay']]   
    mask = round(len(data) * 0.7)
    train = data[:mask]
    test = data[mask:]
    print("train",train.shape)
    print("test",test.shape)

    s = setup(data = train, test_data = test, target = price, fold_strategy = 'timeseries', numeric_features = ['last_updatedYear','last_updatedMonth', 'last_updatedDay'], fold=2, transform_target = True, session_id = 42)
    best = compare_models(sort = 'MAE')
    save_model(best, f'{price}_model')
```


```python
for feature in ["price150", "price300", "price500"]:
    df = get_data()
    train(df, feature)
```

    Transformation Pipeline and Model Succesfully Saved


# Conclusion and What's Next

In this post, we've seen how to build a time series model for forecasting the price of heating oil. In the next post, we'll go over how to deploy these models into a StreamLit application.
We'll also go over how the process on how the data was collected.
