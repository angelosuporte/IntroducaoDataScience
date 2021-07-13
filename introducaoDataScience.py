import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Base de dados, Extração - (https://www.kaggle.com/) Definido variável dados
dados = pd.read_csv('/content/drive/MyDrive/DIO/DadosDoTitanic/train.csv')
dados = dados.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

dados = dados.set_index(['PassengerId'])
dados = dados.rename(columns = {'Survived' : 'target'}, inplace = False)

dados['Sex_F'] = np.where(dados['Sex'] == 'female', 1, 0)

dados['Pclass_1'] = np.where(dados['Pclass'] == 1, 1, 0)
dados['Pclass_2'] = np.where(dados['Pclass'] == 2, 1, 0)
dados['Pclass_3'] = np.where(dados['Pclass'] == 3, 1, 0)

dados = dados.drop(['Pclass', 'Sex'], axis = 1)
dados.head()


dados.fillna(0, inplace= True) 
dados.isnull().sum()

#Amostragem

x_train, x_test, y_train, y_test = train_test_split(dados.drop(['target'], axis = 1),
                                                    dados["target"],
                                                    test_size = 0.3,
                                                    random_state = 1234)

[{'treino': x_train.shape}, {'teste' : x_test.shape}]


rndforest = RandomForestClassifier(n_estimators = 1000,
                                  criterion = 'gini',
                                  max_depth = 5)

rndforest.fit(x_train, y_train)

probabilidade = rndforest.predict_proba(dados.drop('target', axis = 1))[:,1]
classificacao = rndforest.predict(dados.drop('target', axis = 1))

dados['probabilidade'] = probabilidade
dados['classificacao'] = classificacao

dados
