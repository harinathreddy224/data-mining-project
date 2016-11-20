print("Start pre-processing...")

import pandas as pd

df = pd.read_csv('./dataset/sub-set.csv')

print(df)

date = df[["date"]]