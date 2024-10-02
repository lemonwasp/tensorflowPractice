# 레모네이드 판매 예측 (뉴런이 한 개 사용됨)
# 라이브러리 사용
import tensorflow as tf
import pandas as pd
import numpy as np


# 데이터 준비
fileLocation = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
data = pd.read_csv(fileLocation)
data.head()


# 종속변수, 독립변수
independence = data[['온도']]
dependence = data[['판매량']]
print(independence.shape, dependence.shape)


# 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')


# 모델을 학습합니다.
model.fit(independence, dependence, epochs=10000, verbose=0)
model.fit(independence, dependence, epochs=10)


# 모델을 이용합니다.
model.predict(independence)


model.predict(np.array([[15]]))
# 그냥 넣으면 안되고 넘파이 배열로 변환한 다음 넣어줘야 한다.
