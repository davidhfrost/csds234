import pandas as pd

df = pd.read_csv("Data/update.txt", sep="\t", header=None)

print(df.head(100))