import sys
import pandas as pd
from sqlalchemy import create_engine

def extract_category(input_val: str) -> str:
    category, _ = input_val.split('-')
    return category

def category_to_bool(input_val: str) -> int:
    _, value = input_val.split('-')
    return int(value)

def load_data(messages_file, categories_file):
    """Load and merge messages and categories"""
    filepath = 'data/'
    messages = pd.read_csv(filepath + messages_file)
    categories = pd.read_csv(filepath + categories_file)
    df = messages.merge(categories)
    return df

def clean_data(df):
    """Process the data"""
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = categories.iloc[0,:].map(extract_category).values
    categories = categories.applymap(category_to_bool)
    # this field should be binary
    categories.loc[categories['related']==2, 'related'] = categories['related'].mode()[0]
    # only takes value 0 in this dataset
    categories.drop(columns=['child_alone'], inplace=True)
    df = df.drop('categories', axis=1)
    df = df.merge(categories, left_index=True, right_index=True)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """Save clean data to DB"""
    db_path = 'sqlite:///data/{}'.format(database_filename)
    engine = create_engine(db_path)
    df.to_sql('tweets', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_file, categories_file, database_file = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_file, categories_file))
        df = load_data(messages_file, categories_file)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_file))
        save_data(df, database_file)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filenames of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filename of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python data/process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    