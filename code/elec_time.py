import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_excel(io='experimental time.xlsx', header=0)
data = np.array(df)
x = data[:, :3]
x[:, 0] = 1 / x[:, 2]
x[:, 1] = (data[:, 1] != data[:, 6])
x[:, 2] = (data[:, 3] != data[:, 8])
y = data[:, -1]

model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
