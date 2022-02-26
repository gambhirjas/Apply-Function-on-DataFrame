# Apply-Function-on-DataFrame
Apply Function on DataFrame
import pandas as pd
import numpy as np
from datetime import date
import cx_Oracle
import re
import os
from xlsxwriter import Workbook
import math
  def askforinteger(): 
        while True: 
            try: 
                a = int(input("Enter a number: "))
                print("person has entered correct input") 
            except Exception as e : 
                print("there is a error of", e)     
                break 
            finally: 
                        print("close this issue")
                        
import sys
# Get current working directory 
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print ("Current working directory is: {0}".format(cwd))

import perfplot

from faker import Faker

%load_ext line_profiler

import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()

CPU_COUNT


!nvidia-smi


Test Data

def generate_test_data(size: int, days: int = 30):
  fake = Faker()
  Faker.seed(42)

  PRIORITIES = {
    0: 'HIGH',
    1: 'MEDIUM',
    2: 'LOW'
  }

  return pd.DataFrame({
      
    'task_name': [f'Task {i + 1}'
      for i in range(size)],
      
    'due_date': [fake.date_between(start_date='today', end_date=f'+{days}d')
      for _ in range(size)],
      
    'priority': [PRIORITIES[fake.pyint(min_value=0, max_value=(len(PRIORITIES) - 1))]
      for i in range(size)]
      })
      
# Try generate_test_data
tmp_df = generate_test_data(10, 5)
tmp_df.info()
tmp_df.head(10) 

tmp_df['task_name'].loc[0]

tmp_df.info()
tmp_df.head(10)

tmp_df['due_date'].loc[0]
pd.to_datetime(tmp_df['due_date']).dt.date.loc[0]
tmp_df['priority'].loc[0]

from datetime import date
 
# calling the today
# function of date class
today = date.today()
 
print("Today's date is", today)

# Converting the date to the string
Str = date.isoformat(today)
print("String Representation", Str)
print(type(Str))

Generate Test Dataset

K_MAX = 21

# Generate a million rows. Use sample from it to create various size data sets
# Will take some time as Faker will be called sequencially.
test_data_set = generate_test_data(1 + 2 ** K_MAX, 30)

test_data_set.head(5)

def test_data_sample(size: int):
  return test_data_set.sample(n=size).copy().reset_index(drop=True)
  
#Test sample of size 10
test_data_sample(10).head(5)

test_data_set.head(5)
test_data_set.info()

# Optimize DataFrame Storage

Compacting the data not only saves space, but also speeds up the processing. Two common opportunities are:

Converting timestamp strings or datetime to datetime64

Converting strings of enum types to Categorical data type

For fair comparison, compression test data to max extent possible.

import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()

CPU_COUNT

from pandas.api.types import CategoricalDtype

priority_dtype = pd.api.types.CategoricalDtype(
  categories=['LOW', 'MEDIUM', 'HIGH'],
  ordered=True
)
priority_map = dict(enumerate(pd.Series(['']).astype(priority_dtype).cat.categories))
priority_map

test_data_set['priority'] = test_data_set['priority'].astype(priority_dtype)
test_data_set.info()

test_data_set['priority'].loc[0]

Using category for priority reduced the memory usage by ~30%.

Optimize DataFrame Processing Time
The Eisenhower Method

Decide about action needed using Eisenhower method:

Important	Urgent	Binary	Int	Action
True (1)	True (1)	11	3	DO
True (1)	False (0)	10	2	SCHEDULE
False (0)	True (1)	01	1	DELEGATE
False (0)	False (0)	00	0	DELETE

action_dtype = pd.api.types.CategoricalDtype(
  categories=['DELETE', 'DELEGATE', 'SCHEDULE', 'DO'],
  ordered=True
)

action_map = dict(enumerate(pd.Series(['']).astype(action_dtype).cat.categories))
action_map

def eisenhower_action(is_important: bool, is_urgent: bool) -> int:
  return 2 * is_important + is_urgent

def eisenhower_action_str(is_important: bool, is_urgent: bool) -> str:
  return action_map[eisenhower_action(is_important, is_urgent)]
  
import datetime
date = datetime.date.today()
date

# Let's say anything due by tomorrow is Urgent
cutoff_date = datetime.date.today() + datetime.timedelta(days=2)

# Test compute_eisenhower_action

eisenhower_action_str(
  test_data_set.loc[0].priority == 'HIGH',
  test_data_set.loc[0].due_date <= cutoff_date
)
data_sample = test_data_sample(100000)

def loop_impl(df):
  cutoff_date = datetime.date.today() + datetime.timedelta(days=2)

  result = []
  for i in range(len(df)):
    row = df.iloc[i]
    result.append(
      eisenhower_action(row.priority == 'HIGH', row.due_date <= cutoff_date)
    )

  return pd.Series(result)
  
  %timeit data_sample['action_loop'] = loop_impl(data_sample)
  
  def iterrows_impl(df):
  cutoff_date = datetime.date.today() + datetime.timedelta(days=2)
  return pd.Series(
    eisenhower_action(row.priority == 'HIGH', row.due_date <= cutoff_date)
    for index, row in df.iterrows()
  )
  



