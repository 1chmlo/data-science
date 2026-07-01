import duckdb as db
import pandas as pd

pandas_df = pd.read_parquet('../dataset/at_urg_respiratorio_imputado.parquet', engine='fastparquet')
con = db.connect()
con.register('pandas_df', pandas_df)

pandas_df.head()

#query = "select EstablecimientoCodigo, count(*) as total from pandas_df where EstablecimientoCodigo is not null group by EstablecimientoCodigo order by total desc"

query = "select * from pandas_df where EstablecimientoCodigo = 200122"


result = con.execute(query).fetchdf()
print(result)