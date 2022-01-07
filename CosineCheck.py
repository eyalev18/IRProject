import math

import numpy as np
import pandas as pd
import pickle

Q=[0.,0,2]

D = np.zeros((1, 3))
# D=[0.2,0.06]
D = pd.DataFrame(D)
D.loc[0][0] = 2
D.loc[0][1] = 3
D.loc[0][2] = 5
# print(D)
left = sum([a ** 2 for a in Q])
for i in range(len(D)):
    up = sum([a * b for a, b in zip(Q, D.iloc[i])])
    print(up)
    right = sum([a ** 2 for a in D.iloc[i]])
    print(right)
    print(left)
    #sum_down = math.sqrt(left*right)
    sum_down = (left * right) ** 0.5
    print(sum_down)
    total = up / sum_down
    print(total)

