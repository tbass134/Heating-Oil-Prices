
from bs4 import BeautifulSoup
import boto3
import requests
import csv
import re
import json
import os.path
from os import path
import datetime
from datetime import date 
from datetime import timedelta
import codecs
from pathlib import Path

import glob
s3 = boto3.resource('s3')

pattern = '(\$?[0-9]+\.*[0-9]*) per gallon'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
price_data = []
YESTERDAY = datetime.datetime.now() - datetime.timedelta(days=1)

def get_url(url):
  page = requests.get(url, headers=headers)
  if page.ok:
    return page.text
  else:
    print(f'Could not retrieve data for page: {page.url} Got status: {page.status_code}')


def calculate_updated_date(date, date_str):
  if "yesterday" in date_str:
    return date - timedelta(days = 1)
  elif "minutes" in date_str:
    time_diff = [int(s) for s in date_str.split() if s.isdigit()][0]
    return date - timedelta(minutes = time_diff)
  elif "hour" in date_str:
    time_diff = [int(s) for s in date_str.split() if s.isdigit()][0]
    return date - timedelta(hours = time_diff)
  elif "updated just now":
    return date
  else:
    day_diff = [int(s) for s in date_str.split() if s.isdigit()][0]
    return date - timedelta(days = day_diff)

def get_supplier(state, county, div, date):
  supplier = div.find("a", {"class": "pricegridsupplier"})
  if supplier is not None:
    supplier = supplier.text

  last_updated = div.find("div", {"class": "last_updated"})
  if last_updated is not None:
    last_updated = last_updated.text
    last_updated = calculate_updated_date(date, last_updated.strip())
    last_updated = last_updated.isoformat()

  price150 = div.find("span", {"class": "price150"})
  if price150 is not None:
    price150 = price150.text
    price150 = price150.replace('$', '')
    price150 = price150.replace('*', '')

    price150 = re.search(pattern, price150).group(1)

  price300 = div.find("span", {"class": "price300"})
  if price300 is not None:
    price300 = price300.text
    price300 = price300.replace('$', '')
    price300 = price300.replace('*', '')

    price300 = re.search(pattern, price300).group(1)


  price500 = div.find("span", {"class": "price500"})
  if price500 is not None:
    price500 = price500.text
    price500 = price500.replace('$', '')
    price500 = price500.replace('*', '')

    price500 = re.search(pattern, price500).group(1)
    
  return {"state":state,"county":county,"supplier":supplier, "last_updated":last_updated, "price150":price150, "price300":price300, "price500":price500}

def get_prices_by_county(state, county, date):
 
  text = get_url(f'https://www.cheapestoil.com/heating-oil-prices/{county}')
  data = BeautifulSoup(text, 'html.parser')
  mydivs = data.findAll("div", {"class": "col_01 pricegrid"})

  for div in mydivs:
    price_data.append(get_supplier(state, county, div, date))

  mydivsalt = data.findAll("div", {"class": "col_01 pricegridalt"})
  for div in mydivsalt:
    price_data.append(get_supplier(state, county, div, date))
     
def get_data_by_state(state, date):
  print(f'getting data for {state}')
  text = get_url(f'https://www.cheapestoil.com/heating-oil-prices/{state}')
  data = BeautifulSoup(text, 'html.parser')
  counties = data.findAll("div", {"class": "col_13 county_panel"})
  
  for item in counties:
    for a in item.find_all('a', href=True):
      get_prices_by_county(state, a['href'], date)
  pass

def to_s3(data, date):
  object = s3.Object('heating-oil-prices', f'{date.strftime("%m%d%Y%H%M%S")}.json')
  print("saved data to s3")
  object.put(Body=json.dumps(data))

def get_locations(date):
  text = get_url(f'https://www.cheapestoil.com/')
  data = BeautifulSoup(text, 'html.parser')
  options =  data.find("select").findAll("option")
  if len(options) == 0:
    print("could not find items in location selector")
    return

  for option in options:
    if "Connecticut" in option.text:
      value = option.text.replace("Connecticut - ","").replace(" ", "") + "-CT"
    else:
      value = option.text.replace(" ", "") 

    if value != "Connecticut-CT" and value != "Selectyourlocation":
      get_data_by_state(value, date)
  pass


def get_archive_data():
  from datetime import datetime

  for snapshot in glob.glob("archive/website/cheapestoil.com/heating-oil-prices/*/*"):
    print("parsing data from ", snapshot)
    price_data = []
    f=codecs.open(snapshot, 'r', 'utf-8')
    document= BeautifulSoup(f.read(), features="lxml")
    state = snapshot.split("/")[-2]

    date = datetime.strptime(Path(snapshot).stem, '%Y%m%d%H%M%S')
    mydivs = document.findAll("div", {"class": "col_01 pricegrid"})

    for div in mydivs:
        price_data.append(get_supplier(state, None, div, date))

    mydivsalt = document.findAll("div", {"class": "col_01 pricegridalt"})
    for div in mydivsalt:
        price_data.append(get_supplier(state, None, div, date))
      
    print(price_data)
    to_s3(price_data, date)

def lambda_handler(event, context):
  print("*** RUNNING SCRIPT ****")
  date = datetime.datetime.now()
  get_locations(date)
  to_s3(price_data, date)
  
  return {
    'statusCode': 200,
    'body': json.dumps('success')
  }