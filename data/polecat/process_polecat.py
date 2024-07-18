import pandas as pd
import numpy as np
import os

## Extract .txt files in the current directory
txt_files = [f for f in os.listdir() if f.endswith('.txt')]
# Read data into a pandas DataFrame
df = pd.concat([pd.read_csv(f, delimiter='\t') for f in txt_files], ignore_index=True)
# Select relevant columns
selected_columns = [
    'Actor Name', 'Actor Country', 'Recipient Name', 'Recipient Country', 
    'Event Date', 'Event Type', 'Quad Code'
]
# Extract the relevant columns
df = df[selected_columns]
# Split the 'Actor Country' and 'Recipient Country' columns by ';'
df['Actor Country'] = df['Actor Country'].str.split('; ')
df['Recipient Country'] = df['Recipient Country'].str.split('; ')
# Explode the 'Actor Country' and 'Recipient Country' columns JOINTLY
df = df.explode('Actor Country').explode('Recipient Country')
# Remove rows where 'Actor Country' and 'Recipient Country' are the same country
df = df[df['Actor Country'] != df['Recipient Country']]
# Remove rows where 'Actor Country' or 'Recipient Country' are None (not na)
df = df[(df['Actor Country'] != 'None') & (df['Recipient Country'] != 'None')]
# Select relevant columns
selected_columns = [
    'Actor Country', 'Recipient Country', 'Event Date', 'Event Type', 'Quad Code'
]
# Extract the relevant columns and remove NAs
df = df[selected_columns].dropna()
# Transform 'Event Date' into 'Event Month'
df['Event Month'] = pd.to_datetime(df['Event Date']).dt.to_period('M')
# Map unique pairs of year-month in 'Event Month' to integers
map_month = {v: i for i, v in enumerate(np.sort(df['Event Month'].unique()))}
df['Event Month Id'] = df['Event Month'].map(map_month)
# Map unique 'Event Type' to integers
map_event_type = {v: i for i, v in enumerate(np.sort(df['Event Type'].unique()))}
df['Event Type Id'] = df['Event Type'].map(map_event_type)
# Map unique 'Quad Code' to integers
map_quad_code = {v: i for i, v in enumerate(np.sort(df['Quad Code'].unique()))}
df['Quad Code Id'] = df['Quad Code'].map(map_quad_code)
# Map unique 'Actor Country' and 'Recipient Country' to integers (from a unique mapping list for both)
countries = np.sort(df['Actor Country'].unique())
countries = np.sort(np.unique(np.concatenate([countries, df['Recipient Country'].unique()])))
map_country = {v: i for i, v in enumerate(countries)}
df['Actor Country Id'] = df['Actor Country'].map(map_country)
df['Recipient Country Id'] = df['Recipient Country'].map(map_country)
# Sort by 'Event Month Id'
df = df.sort_values('Event Month Id')

# Save the DataFrame to a CSV file
df.to_csv('polecat.csv', index=False)
# Save all map files
pd.Series(map_month).to_csv('map_month.csv', header=False)
pd.Series(map_event_type).to_csv('map_event_type.csv', header=False)
pd.Series(map_quad_code).to_csv('map_quad_code.csv', header=False)
pd.Series(map_country).to_csv('map_country.csv', header=False)
