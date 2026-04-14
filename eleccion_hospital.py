import duckdb as db
import pandas as pd

pandas_df = pd.read_parquet('./dataset/dataset.parquet', engine='fastparquet')
con = db.connect()
con.register('pandas_df', pandas_df)

pandas_df.head()

query = "select EstablecimientoGlosa, count(*) as total from pandas_df group by EstablecimientoGlosa order by total desc limit 10"

result = con.execute(query).fetchdf()
print(result)