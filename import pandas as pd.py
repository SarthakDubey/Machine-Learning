import pandas as pd
import numpy as np
import math
from random import randint


df = pd.read_csv("classasgntrain1.dat", delim_whitespace=True, header=None)

new1 = pd.concat([df[0], df[1]], axis=1, keys=['x1', 'x2'])  # class0
new2 = pd.concat([df[2], df[3]], axis=1, keys=['x1', 'x2'])  # class1

new1['class'] = 0
new2['class'] = 1

xAxisList = df[0].values.tolist() + df[2].values.tolist()
yAxisList = df[1].values.tolist() + df[3].values.tolist()

x_0 = min(xAxisList)
x_1 = max(xAxisList)
y_0 = min(yAxisList)
y_1 = max(yAxisList)


new_df = new1.append(new2, ignore_index=True)


N = len(new_df)
X = np.array(new_df[['x1', 'x2']])
y = np.array(new_df[['class']])

k = 5

error = 0

for i in range(X.shape[0]):
    x1 = X[i][0]
    y1 = X[i][1]
    dtrack = pd.DataFrame()
    d_arr = []
    carr = []

    for j in range(X.shape[0]):
        if i != j:
            x2 = X[j][0]
            y2 = X[j][1]
            distance = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
            d_arr.append(distance)
            carr.append(y[j][0])

    dtrack['distance'] = d_arr
    dtrack['class'] = carr

    dtrack = dtrack.sort_values('distance')
    dtrack = dtrack.set_index(np.arange(len(dtrack.index)))

    sum = 0
    for m in range(k):
        print dtrack['distance'][m]
        sum = sum + dtrack['class'][m]

    df = sum/float(k)

    if df < 0.5 and y[i][0] != 0 or df > 0.5 and y[i][0] != 1:
        error = error + 1


error = error/float(X.shape[0])
print error


def genData(classType, value):
    xElement = []
    m0 = [[-0.132, 0.320, 1.672, 2.230,  1.217, -0.819,  3.629,  0.8210,  1.808, 0.1700], [-0.711, -1.726, 0.139, 1.151,
          -0.373, -1.573, -0.243, -0.5220, -0.511, 0.5330]]
    m1 = [[-1.169, 0.813, -0.859, -0.608, -0.832, 2.015, 0.173, 1.432,  0.743, 1.0328], [2.065, 2.441,  0.247, 1.806,
          1.286, 0.928, 1.923, 0.1299, 1.847, -0.052]]
    m0 = np.array(m0)
    m1 = np.array(m1)
    for j in range(value):
        r = randint(0, 9)
        idx = math.floor(r)
        idx = int(idx)
        if classType == 0:
            m = m0[:, [idx]]
        else:
            m = m1[:, [idx]]
        r1 = np.random.randn(2, 1)
        r2 = r1/float(math.sqrt(5))
        element = m+r2
        xElement.append([element[0][0], element[1][0]])
    return xElement