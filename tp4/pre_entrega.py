import pandas as pd



path= "./data/europe.csv"

df= pd.read_csv(path)

print(df.head())