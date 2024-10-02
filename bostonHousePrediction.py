# 라이브러리 사용
import tensorflow as tf
import pandas as pd
import numpy as np


# 1. 과거의 데이터를 준비합니다.
fileLocation = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
data = pd.read_csv(fileLocation)
print(data.columns)
data.head()


# 독립변수, 종속변수 분리
independence = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
dependence = data[['medv']]
print(independence.shape, dependence.shape)


# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')


# 3. 데이터로 모델을 학습(FIT)합니다.
model.fit(independence, dependence, epochs=1000, verbose=0)
model.fit(independence, dependence, epochs=10)


# 4. 모델을 이용합니다.
model.predict(independence[5:10])
dependence[5:10]


# 모델의 수식 확인
model.get_weights()
