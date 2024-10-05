import pandas as pd


fileLocation = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
data = pd.read_csv(fileLocation)
data.head()


# 원핫인코딩
encode = pd.get_dummies(data)
encode.head()


print(data.dtypes)


# 품종 타입을 범주형으로 바꾸어 준다.
data['품종'] = data['품종'].astype('category')
print(data.dtypes)


# NA값을 체크해 봅시다.
data.isna().sum()


data.tail()


# NA값에 꽃잎폭 평균값을 넣어주는 방법
mean = data['꽃잎폭'].mean()
print(mean)
data['꽃잎폭'] = data['꽃잎폭'].fillna(mean)
data.tail()
