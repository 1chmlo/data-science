import duckdb as db
import pandas as pd

pandas_df = pd.read_parquet('./dataset/dataset.parquet', engine='fastparquet')
con = db.connect()
con.register('pandas_df', pandas_df)

pandas_df.head()

query = "select Cuasa, count(*) as total from pandas_df group by Cuasa order by total desc"

result = con.execute(query).fetchdf()
print(result)