# %% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% read file
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()
data.describe()
data.plot(kind='scatter', x='Population', y='Profit')

# %% 初始化变量
data.insert(0, 'Ones', 1)
# Parameters
# locint - Insertion index. Must verify 0 <= loc <= len(columns).
# column - str, number, or hashable object - Label of the inserted column.
# value - int, Series, or array-like
# allow_duplicates - bool, optional


def get_Xy(df):
    tempX = df.iloc[:, :-1].values
    tempy = df.iloc[:, [-1]].values
    # -1 如果不带方括号，iloc就会返回 Series 而不是 Dataframe，
    # 如果返回了Series，则还需要用 Series.to_frame().values。
    # 为了确保 iloc 返回的是 dataframe，只需传给它的是列表即可。
    return tempX, tempy

# 初始化 theta 向量


def init_theta(df):
    return np.zeros([df.shape[1], 1])


X, y = get_Xy(data)
theta = init_theta(X)
# %% 计算 J


def computeCost(X, y, theta):
    inner = np.power((X @ theta - y), 2)
    return np.sum(inner)/(2*len(X))
# 无论是数组还是矩阵，
# 点乘均使用 np.multiply
# 矩阵乘法使用 @


def gradientDescent(X, y, theta, alpha, iters):
    # 返回最终的 theta 和 J_history 值
    m = len(y)
    J_history = np.zeros([iters, 1])
    for iter in range(iters):
        theta = theta - X.T @ (X @ theta - y)*alpha/m
        J_history[iter] = computeCost(X, y, theta)
        # print('J is: ', J_history[iter])
    return theta, J_history


def normalize_features(df):
    return df.apply(lambda column: (column-column.mean())/column.std(), axis=0)
    # axis{0 or ‘index’, 1 or ‘columns’}, default 0
    # Axis along which the function is applied:
    #   0 or ‘index’: apply function to each column.
    #   1 or ‘columns’: apply function to each row.


# %% 试一下

alpha = 0.01
iters = 1500
g, cost = gradientDescent(X, y, theta, alpha, iters)
computeCost(X, y, g)
computeCost(X, y, [[-1], [2]])

# %% 画出来

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[1, 0] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# %%
plt.plot(range(iters), cost)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()

# %% Ex1_multi

raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
raw_data.head()

# 泛化
data2 = normalize_features(raw_data)
data2.head()
