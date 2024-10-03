# 라이브러리 사용
import tensorflow as tf
import pandas as pd


# 1.과거의 데이터를 준비합니다.
fileLocation = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
data = pd.read_csv(fileLocation)
data.head()


# 원핫인코딩
encode = pd.get_dummies(data)
encode = encode.astype(int)
encode.head()


print(encode.columns)


# 독립변수, 종속변수
independence = encode[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependence = encode[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(independence.shape, dependence.shape)


# 2.모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


# 3.데이터로 모델을 학습(FIT)합니다.
model.fit(independence, dependence, epochs=300)


# 모델을 이용합니다.
model.predict(independence[0:5])


print(dependence[-5:])


# 학습한 가중치
model.get_weights()
