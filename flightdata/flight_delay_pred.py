#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:07:38 2020
@author: alphonse
"""
import pandas as pd
flight_data = pd.read_csv('flightdata_orig.csv')

flight_data = flight_data.drop('Unnamed: 25', axis = 1)

features = ["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]
flight_data = flight_data[features] 

flight_data = flight_data.fillna({'ARR_DEL15': 1}) 
flight_data.iloc[177:185]

import math 
for index, row in flight_data.iterrows(): 
    flight_data.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100) 
flight_data.head()

flight_data = pd.get_dummies(flight_data, columns=['ORIGIN', 'DEST'])
flight_data.head()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(flight_data.drop('ARR_DEL15', axis=1), flight_data['ARR_DEL15'], test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(random_state=13) 
model.fit(train_x, train_y)

predicted = model.predict(test_x) 
model.score(test_x, test_y)

#predict_delay function
# departure_date_time = input('Enter Date and Time')
# origin = input('Enter Orign')
# destination = input('Enter Dest')
# print(departure_date_time)
# print(type(departure_date_time))
    
def predict_delay(departure_date_time, origin, destination): 

    from datetime import datetime 

    try: 
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S') 
    except ValueError as e: 
        return 'Error parsing date/time - {}'.format(e) 

    month = departure_date_time_parsed.month 
    day = departure_date_time_parsed.day 
    day_of_week = departure_date_time_parsed.isoweekday() 
    hour = departure_date_time_parsed.hour 

    origin = origin.upper() 
    destination = destination.upper() 

    input = [{'MONTH': month, 
            'DAY': day, 
            'DAY_OF_WEEK': day_of_week, 
            'CRS_DEP_TIME': hour, 
            'ORIGIN_ATL': 1 if origin == 'ATL' else 0, 
            'ORIGIN_DTW': 1 if origin == 'DTW' else 0, 
            'ORIGIN_JFK': 1 if origin == 'JFK' else 0, 
            'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
            'ORIGIN_SEA': 1 if origin == 'SEA' else 0, 
            'DEST_ATL': 1 if destination == 'ATL' else 0, 
            'DEST_DTW': 1 if destination == 'DTW' else 0, 
            'DEST_JFK': 1 if destination == 'JFK' else 0, 
            'DEST_MSP': 1 if destination == 'MSP' else 0, 
            'DEST_SEA': 1 if destination == 'SEA' else 0 }] 

    return model.predict_proba(pd.DataFrame(input))[0][0]


