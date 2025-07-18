from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

def data_pull(file_in, file_out):
    load_dotenv()
    server = os.getenv('SERVER')
    database = os.getenv('DATABASE')

    if not all([server, database]):
        raise ValueError('Missing environment variables')

    connection_string = (
        f"mssql+pyodbc://@{server}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
    )

    try:
        engine = create_engine(connection_string)
        print('Connection successful')
    except Exception as e:
        print(f'Connection failed: {e}')
        return

    with open(file_in, 'r', encoding='utf-8') as file:
        query = file.read()

    df = pd.read_sql(query, con=engine)
    df.to_csv(file_out, index=False, encoding='utf-8-sig')
    print(f'{file_in} pulled successfully')
    engine.dispose()

if __name__ == '__main__':
    raw_in = r'../sql/RAW_LOAD.sql'
    raw_out = r'../../data/backup/raw_set.csv'
    data_pull(file_in=raw_in, file_out=raw_out)

    oob_in = r'../sql/OOB_LOAD.sql'
    oob_out = r'../../data/backup/oob_set.csv'
    data_pull(file_in=oob_in, file_out=oob_out)