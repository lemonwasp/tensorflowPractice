# pandas라이브러리 import
import pandas as pd


# 파일들로부터 데이터 읽어오기
fileLocation = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(fileLocation)

fileLoaction = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(fileLocation)

fileLocation = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(fileLocation)


# 데이터 모양으로 확인하기
print(lemonade.shape)
print(boston.shape)
print(iris.shape)


# 칼럼이름 출력
print(lemonade.columns)
print(boston.columns)
print(iris.columns)


# 데이터의 독립변수와 종속변수를 분리
independence = lemonade[['온도']]
dependence = lemonade[['판매량']]
print(independence.shape, dependence.shape)

independence = boston[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat','medv']]
dependence = boston[['medv']]
print(independence.shape, dependence.shape)

independence = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependence = iris[['품종']]
print(independence.shpae, dependence.shape)


lemonade.head()
boston.head()
iris.head()
