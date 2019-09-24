# Dublin Bikes Analysis

## About 

This repository contains the code and data to accompany my Medium blog post titled How I used Machine Leaning to improve my Dublin Bikes transit. In this project I analyse Dublin Bikes data which is made freely available by the Irish national open data project and was collected and shared by [James Lawlor](https://github.com/jameslawlor/dublin-bikes-timeseries-analysis). 

## Usage

The analysis has been split into 3 notebooks that roughly correspond to the 3 questions discussed in the article. 
1. `data_load_processing_viz.ipynb` pre-processes the messy input data files, makes some overview visualisations and saves the processed data as `full_data.csv` which is used for all further analysis. 
2. `timeseries_analysis.ipynb` runs a time series analysis on the city wide available bikes fitting a SARIMAX model. 
3. `get_nearest_available_bike.py` will find the soonest available bike for a given input location. The script can be run with the following syntax from the command line: `python get_nearest_available_bike.py 53.345761 -6.235113 `. `get_nearest_available_bike.ipynb` is the same script in jupyter notebook format. Additionally, users will need to add their own google maps and dublin bikes API keys in `config.json`.




