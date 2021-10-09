import sys
import numpy as np
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    df_mes = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)
    df = df_mes.merge(df_cat, on='id', how='inner')
    return df


def clean_data(df):
    df_drop=df.drop_duplicates(subset=['id'])
    return df_drop


def save_data(df, database_filename):
    conn = sqlite3.connect(database_filename)
    df.to_sql('message', con=conn)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()