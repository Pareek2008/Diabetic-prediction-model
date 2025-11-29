import pandas as pd

df = pd.read_csv("data/diabetes.csv", sep="\t", engine="python")
print(df.columns.tolist())
