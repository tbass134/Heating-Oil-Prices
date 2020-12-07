
from bs4 import BeautifulSoup
import requests
import csv
import re
import json
import os.path
from os import path
import datetime
from datetime import date 
from datetime import timedelta

  

import airflow
from airflow.operators import python_operator

from airflow.hooks.postgres_hook import PostgresHook
pg_hook = PostgresHook(postgres_conn_id='postgres_default')
pattern = '(\$?[0-9]+\.*[0-9]*) per gallon'


now = datetime.datetime.now()

YESTERDAY = datetime.datetime.now() - datetime.timedelta(days=1)

def calculate_updated_date(date_str):
  if "yesterday" in date_str:
    return now - timedelta(days = 1)
  elif "minutes" in date_str:
    time_diff = [int(s) for s in date_str.split() if s.isdigit()][0]
    return now - timedelta(minutes = time_diff)
  elif "hour" in date_str:
    time_diff = [int(s) for s in date_str.split() if s.isdigit()][0]
    return now - timedelta(hours = time_diff)
  else:
    day_diff = [int(s) for s in date_str.split() if s.isdigit()][0]
    return now - timedelta(days = day_diff)


def get_supplier(div):
  supplier = div.find("a", {"class": "pricegridsupplier"})
  if supplier is not None:
    supplier = supplier.text

  last_updated = div.find("div", {"class": "last_updated"})
  if last_updated is not None:
    last_updated = last_updated.text
    last_updated = calculate_updated_date(last_updated)

  price150 = div.find("span", {"class": "price150"})
  if price150 is not None:
    price150 = price150.text
    price150 = price150.replace('$', '')
    price150 = re.search(pattern, price150).group(1)

  price300 = div.find("span", {"class": "price300"})
  if price300 is not None:
    price300 = price300.text
    price300 = price300.replace('$', '')
    price300 = re.search(pattern, price300).group(1)


  price500 = div.find("span", {"class": "price500"})
  if price500 is not None:
    price500 = price500.text
    price500 = price500.replace('$', '')

    price500 = re.search(pattern, price500).group(1)
    
  return {"supplier":supplier, "last_updated":last_updated, "price150":price150, "price300":price300, "price500":price500}



def run():
  print("*** RUNNING SCRIPT ****")
  page = requests.get("https://www.cheapestoil.com/heating-oil-prices/DutchessCounty")
  page_data = BeautifulSoup(page.text, 'html.parser')
  

  now = datetime.datetime.now()
  today = date.today() 

  mydivs = page_data.findAll("div", {"class": "col_01 pricegrid"})
  for div in mydivs:
    data = get_supplier(div)

    dts_insert = "insert into prices (company, date, price_150, price_300, price_500) values (%s, %s, %s, %s, %s)"
    print("dts_insert",dts_insert)
    pg_hook.run(dts_insert, parameters=(data["supplier"], data["last_updated"], data["price150"], data["price300"], data["price500"]))


  mydivsalt = page_data.findAll("div", {"class": "col_01 pricegridalt"})
  for div in mydivsalt:
    data = get_supplier(div)
    
    dts_insert = "insert into prices (company, date, price_150, price_300, price_500) values (%s, %s, %s, %s, %s)"
    print("dts_insert",dts_insert)
    pg_hook.run(dts_insert, parameters=(data["supplier"], data["last_updated"], data["price150"], data["price300"], data["price500"]))



default_args = {
    'owner': 'Composer Example',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': YESTERDAY,
}

with airflow.DAG('get_heating_price_dag','catchup=False',default_args=default_args,schedule_interval='0 * * * *') as dag:
    get_prices_dag_run_conf = python_operator.PythonOperator(
        task_id='get_prices', python_callable=run)



