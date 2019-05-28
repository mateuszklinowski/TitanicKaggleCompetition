import numpy as np
import pandas as pd
import matplotlib as mpl

titanic_df = pd.read_csv('data/prepared_all.csv',',')

print(titanic_df.head(20))
print(titanic_df.describe())
