import pandas as pd
import numpy as np

filename = "./data/data.csv"
df = pd.read_csv(filename)
cols = df.columns

cols_main = []
for col in cols:
    print(col)
    print(f"{list(df[col].unique())}")
    