import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/findff3.csv', low_memory=False)

df2 = df[df['Career Minutes'] != 0]
df2 = df2[df2['Season Minutes'] != 0]

df2.to_csv('C:/Users/aurim/Desktop/Mokslai/findff3_n0c_s.csv', index=False)

exp = df2.describe()
exp.to_csv('C:/Users/aurim/Desktop/Mokslai/exp_fin_n0c_s.csv', index=False)


















