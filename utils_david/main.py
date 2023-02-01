from flow import Flow
from pathlib import Path
import pandas as pd

ROOT = '/home/hackathon_2023/'

def get_data(path=None, limit=-1):
    if not path:
        path = Path(ROOT) / 'data' / 'enron' / 'emails.parquet'
    df = pd.read_parquet(path=path, engine='pyarrow')
    return df

if __name__ == '__main__':
    mails = get_data()
    test_mail = mails.iloc[100]
    my_flow = Flow("Enron_investigation")




