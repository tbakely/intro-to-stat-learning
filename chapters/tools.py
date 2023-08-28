import psycopg2
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def execute_statement(sql: str):
    with psycopg2.connect(
        host="localhost", database="thefantasybot", user="tbakely"
    ) as conn:
        df = pd.read_sql(sql, conn)
        return df
    

