# y,theta 用 一维array 结构，shape = (97,)

# %% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf

# %% read file
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()
data.describe()
data.plot(kind='scatter', x='Population', y='Profit')

# %% 初始化变量


def get_Xy(df):
    df.insert(0, 'Ones', 1)
    # Parameters
    # locint - Insertion index. Must verify 0 <= loc <= len(columns).
    # column - str, number, or hashable object - Label of the inserted column.
    # value - int, Series, or array-like
    # allow_duplicates - bool, optional
    tempX = df.iloc[:, :-1].values
    tempy = np.array(df.iloc[:, -1])
    return tempX, tempy

# 初始化 theta 向量


def init_theta(df):
    return np.zeros(df.shape[1])


X, y = get_Xy(data)
theta = init_theta(X)
# %% 计算 J


def computeCost(_X, _y, _theta):
    inner = np.power((_X @ _theta - _y), 2)
    return np.sum(inner)/(2*len(_X))
# 无论是数组还是矩阵，
# 点乘均使用 np.multiply
# 矩阵乘法使用 @


def gradientDescent(_X, _y, _theta, _alpha, _iters):
    # 返回最终的 theta 和 J_history 值
    m = len(_y)
    J_history = np.zeros([_iters, 1])
    for iter in range(_iters):
        _theta = _theta - _X.T @ (_X @ _theta - _y)*_alpha/m
        J_history[iter] = computeCost(_X, _y, _theta)
        # print('J is: ', J_history[iter])
    return _theta, J_history


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
computeCost(X, y, [-1, 2])

# %% 画出来

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0] + g[1] * x

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

X2, y2 = get_Xy(data2)
print(X2.shape, type(X2))
print(y2.shape, type(y2))

alpha = 0.01
theta = init_theta(X2)
iters = 500
final_theta, cost2 = gradientDescent(X2, y2, theta, alpha, iters)

plt.plot(range(iters), cost2)
plt.xlabel('iters')
plt.ylabel('cost')
plt.show()
print(final_theta)

# %%
# 关于可变对象作为方法参数传递时


def change(_a):
    # _a.append(2) # a 变了
    # _a[0]=_a[0]+1  # a 变了
    # _a[1]=_a[1]+1
    # 以上貌似属于改变这个name所对应的内存地址中的值，所以外面a跟着变
    # _a = _a + [2] # a 不变
    # _a = [i+1 for i in _a ]  # a 不变
    _a = [0, 0, 0]
    # 以上貌似属于给这个name赋予一个新值（内存地址），所以外面a不变
    print('函数内 _a = ', _a, id(_a))


a = [1, 2]
change(a)
print('执行后 a = ', a, id(a))
