import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import sys

from PIL import Image
from io import BytesIO
from mpl_toolkits.basemap import Basemap
from pandas.io.json import json_normalize
from datetime import datetime, timedelta

# load preprocessed data from data_load_processing_viz.ipynb
full_data = pd.read_csv('full_data.csv', parse_dates=['DateTime'])
station_locations = pd.read_csv('station_locations.csv')

# get values from config file
with open('config.json') as config:
    config = json.load(config)

google_api_key = config['api_keys']['google_maps']
dublin_bikes_api_key = config['api_keys']['dublin_bikes']

bikes_base = config['hosts']['dublin_bikes']
dist_matrix_base = config['hosts']['google_distance_matrix']
directions_base = config['hosts']['google_directions']
staticmap_base = config['hosts']['google_static_map']

# parameters
num_time_intervals = 10  # how many 2 min intervals to consider in historical windows
n = 0  # returns the nth nearest bike station


# helper functions 
def get_lambda(data, start_timestamp, periods):
    """
    Given a date and a number of time periods to consider, calculate the historical estimate
    for rate parameter lambda (events/time) for the data
    """
    end_timestamp = start_timestamp + timedelta(minutes=2*periods)
    subset_day = data[data.index.dayofweek == start_timestamp.dayofweek]
    subset_time = subset_day.between_time(start_timestamp.time(), end_timestamp.time())
    lambda_vals = subset_time.mean(axis=0)
    return lambda_vals

def calculate_expected_waiting_time(lambda_vals, periods):
    """
    Convert lambda vector into vector of expected waiting times according to
    the exponential distribution
    """
    expected_vals = 1/lambda_vals
    expected_vals[expected_vals == np.inf] = periods  # heuristic
    expected_vals[expected_vals == 1] = 0  # heuristic
    return expected_vals

def parse_coords(coords):
    latitude = coords[0]
    longitude = coords[1]
    return '{},{}'.format(str(latitude), str(longitude))


def to_formatted_string(latlon_df):
    """Convert DataFrame of lat-lons to google api string format"""
    coords_series = latlon_df[['Latitude', 'Longitude']].apply(parse_coords, axis=1)
    formatted_string = coords_series.str.cat(sep='|')
    return formatted_string

def format_coords(data, max_locations=25):
    """
    Format DataFrame containing latitude and longitude values to send to googles
    distance matrix api and account for upper limit of coordinates per call.
    
    See: https://developers.google.com/maps/documentation/distance-matrix/intro
    """
    num_chunks = data.shape[0]//max_locations
    split_data = np.array_split(data, num_chunks)
    list_of_formatted_coords = [to_formatted_string(data) for data in split_data]
    return list_of_formatted_coords

def extract_durations(resp_json_payload):
    """Parse durations from google distance matrix output"""
    output_df = json_normalize(resp_json_payload['rows'][0]['elements'])
    durations = output_df['duration.value']
    return durations

def call_api(url):
    response = requests.get(url)
    resp_json_payload = response.json()
    return resp_json_payload

def make_url(base, **kwargs):
    """format URL for api calls"""
    args = ('&'.join(['%s=%s' % (key, value) for (key, value) in kwargs.items()]))
    return base + args

if __name__ == '__main__':
    
    origins = parse_coords((float(sys.argv[1]), float(sys.argv[2])))
    # origins = '53.344389,-6.239678'  # uncomment for bord gais energy theatre 

    # get live station info from dublin bikes
    print('Loading live dublin bikes status...')
    url = make_url(bikes_base, apiKey=dublin_bikes_api_key)
    live_station_info = call_api(url)
    
    live_info = json_normalize(live_station_info)

    # subset based on the stations in our data (note: 4 stations from our historical data no longer exist)
    live_info_reduced = live_info[live_info['name'].isin(list(station_locations['Name']))]
    live_info_reduced = live_info_reduced[['name', 'available_bikes']]

    # we will define any station with less than 2 bikes as empty
    live_info_reduced['empty'] = live_info_reduced['available_bikes'] < 2
    index = [station.replace(' ', '_') for station in live_info_reduced['name']]
    live_info_reduced.index = index
    
    # preprocess data
    print('Calculating historical waiting times...')
    full_data.index = full_data['DateTime']
    station_columns = [column_name for column_name in list(full_data.columns) if column_name.isupper()]
    stations_df = full_data[station_columns]
    stations_df_bool = stations_df.astype('bool') * 1
    
    # calculate expected waiting times
    timestamp = pd.Timestamp.now()  
    lambda_vals = get_lambda(stations_df_bool, timestamp, num_time_intervals)
    waiting_times = calculate_expected_waiting_time(lambda_vals, num_time_intervals)
    
    # combine outputs to waiting times for empty stations
    waiting_times.name = 'waiting_times'
    stations_status = pd.concat([live_info_reduced, waiting_times/2], axis=1, sort=True)
    stations_status['name'] = stations_status['name'].fillna(stations_status.index.to_series())
    stations_status['empty'] = stations_status['empty'].fillna(True)  # assume missing stations are empty
    stations_status['waiting_time_if_empty'] = stations_status['empty'] * stations_status['waiting_times']

    # getting durations to all stations using google distance matrix api main logic
    print('Calculating walking distances to all bike stations...')
    station_locations.sort_values(by='Name', inplace=True)
    destinations = format_coords(station_locations)
    durations = []
    for destinations_chunk in destinations:
        url = make_url(dist_matrix_base, mode='walking', origins=origins, destinations=destinations_chunk, key=google_api_key)
        response_json = call_api(url)
        durations_chunk = extract_durations(response_json)
        durations.extend(durations_chunk)

        
    station_locations['duration'] = [duration/60 for duration in durations]
    station_locations['Name'] = station_locations['Name'].str.replace(' ', '_')
    station_locations['Name'] = station_locations['Name'].replace({'CHATHAM_STREET': 'CHATHAM_STREET/CLARENDON_ROW'})
    
    # merge waiting times with travel times to obtain total time
    merged_df = pd.merge(stations_status, station_locations, left_index=True, right_on='Name', how='inner')
    merged_df['total_time'] = merged_df['waiting_time_if_empty'] + merged_df['duration']
    summary_df = merged_df[['Name', 'Latitude', 'Longitude', 'duration', 'waiting_time_if_empty', 'total_time']].sort_values(by='total_time').reset_index(drop=True)

    # ordered df of fastest locations to get a bike
    print('Five nearest stations with waiting times:')
    summary_df.head()
    
    print('Plotting route to nearest available bike...')
    # get directions to a nth station 
    latitude = summary_df.loc[n,'Latitude']
    longitude = summary_df.loc[n,'Longitude']
    destination = summary_df.loc[n, 'Name'].title().replace('_', ' ')
    duration = np.round(summary_df.loc[n,'duration'], 2)
    waiting_time_if_empty = np.round(summary_df.loc[n,'waiting_time_if_empty'], 2)
    total_time = np.round(summary_df.loc[n,'total_time'], 2)
    
    # call google directions api
    url = make_url(directions_base, origin=origins, destination=parse_coords((latitude, longitude)), 
            mode='walking', key=google_api_key)
    directions_json = call_api(url)
    
    # parse output
    directions_df = json_normalize(directions_json['routes'][0]['legs'][0]['steps'])
    directions_df = directions_df[['start_location.lat', 'start_location.lng']].append(
        {'start_location.lat':latitude, 'start_location.lng':longitude}, ignore_index=True)

    # plot using googles own static map api
    path = to_formatted_string(directions_df.rename(columns={'start_location.lat':'Latitude', 'start_location.lng':'Longitude'}))
    coords = '|'.join([origins, parse_coords((latitude, longitude))])
    url = make_url(staticmap_base, markers=coords, size='640x640', scale='2', path=path, key=google_api_key)
    
    respones_img = requests.get(url)   
    im = Image.open(BytesIO(respones_img.content))
    
    fig, ax = plt.subplots(figsize=(12, 12))

    plt.imshow(im)
    label = '{}\nWalk time: {}\nWaiting time: {}\nTotal: {}'.format(destination, duration, waiting_time_if_empty, total_time)
    plt.text(0.15, 0.915, label,
         horizontalalignment='center',
         verticalalignment='center', fontsize=16,
         transform = ax.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='white'))
    plt.axis('off')
    plt.show()
    