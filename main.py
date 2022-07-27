import pandas
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

data = pandas.read_csv('./train.csv', delimiter=(','))
data = data[pandas.isnull(data['Age']) == 0 ]
data = data[pandas.isnull(data['Fare']) == 0 ]
data = data[pandas.isnull(data['Parch']) == 0 ]

Y = data['Survived']
X = pandas.get_dummies(data.loc[:, ['Age', 'Fare', 'Parch']])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
logRegModel = LogisticRegression(max_iter=1000)
logRegModel.fit(X_train, Y_train)
#print('LogisticRegression Score: ', logRegModel.score(X_test,Y_test))

predictions = logRegModel.predict_proba(X_test)
fpr, tpr, thres = roc_curve(Y_test, predictions[:, 1] )
plt.plot( fpr, tpr, label = 'All columns' )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Dummy data ROC curve')
plt.legend(loc = 0)
plt.grid()
plt.show()



data = pandas.read_csv('./train.csv', delimiter=(','))
labelEncoder = LabelEncoder()
labelEncoder.fit(data['Sex'])
transformedGender = pandas.Series(data = labelEncoder.transform(data['Sex']))
data['Sex'] = transformedGender

# Проверяем процент данных которые будут потеряны в случае удаления пустых значений
c_data = data
c_data = c_data[pandas.isnull(c_data['Age']) == 0]
c_data = c_data[pandas.isnull(c_data['Embarked']) == 0]
del_perc = 100 - c_data.shape[0] / (data.shape[0] / 100) 
print("Percent will be deleted: ", del_perc)

# Заполняем пропуски средним значением
ages = data['Age'][pandas.isnull(data['Age']) == 0]
mid_age = math.ceil(ages.sum() / ages.shape[0])
data['Age'][pandas.isnull(data['Age']) == 1] = mid_age

# Добавляем для пустых полей индикатор
data['Embarked_empty'] = 0
data['Embarked_empty'][pandas.isnull(data['Embarked']) == 1] = 1

# Отсекаем признаки логически не имеющие корелляционную зависимость к выживаемости, колличественные переменные,
# оставляем категориальные переменные которые могли влиять на этот показатель.
X = pandas.get_dummies(data.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'Embarked_empty']])

Y = data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
logRegModel = LogisticRegression(max_iter=1000)
logRegModel.fit(X_train, Y_train)
#print('LogisticRegression Score: ', logRegModel.score(X_test,Y_test))

predictions = logRegModel.predict_proba(X_test)
fpr, tpr, thres = roc_curve(Y_test, predictions[:, 1] )
plt.plot( fpr, tpr, label = 'All columns' )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('EDA data ROC curve')
plt.legend(loc = 0)
plt.grid()
plt.show()


