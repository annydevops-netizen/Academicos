import pandas as pd 
from sklearn.model_selection import train_test_split # para treino e teste embaralhamento etc
from sklearn.preprocessing import StandardScaler # para normalização dos dados
from sklearn.linear_model import Perceptron # Importando o modelo de Perceptron
from sklearn.metrics import accuracy_score # para avaliar a acurácia do modelo, comparar as previsões com os rótulos reais

dataframe = pd.read_csv('Data/titanic.csv')

#Pegar colunas relevantes: Survived = (y) e Pclass, Age, Fare = (X)

df_IsolandoDados = dataframe[['Survived','Pclass','Age','Fare']]

#Como Age contem NaN, preencher com a média foi o indicado

df_IsolandoDados['Age']= df_IsolandoDados['Age'].fillna(df_IsolandoDados['Age'].mean())

print(df_IsolandoDados)

#Problemas com NaN resolvidos com média e conferido com isnull().sum() e isna()
# axis = 0 → operar no eixo das linhas (indice)
# axis = 1 → operar no eixo das colunas
# axis=0 → ↓
# axis=1 → →


#Separar X e Y

y = df_IsolandoDados['Survived']
x = df_IsolandoDados.drop('Survived', axis=1)

#Treino e teste

y_train, y_test, x_train, x_test = train_test_split(
    y,
    x,
    test_size=0.2, random_state=42
)

#Normalização de dos valores 
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Começando a treinar o modelo de Perceptron
modelo = Perceptron(
    max_iter=1000, # Maximum Iterations, maximo de interações/ épocas
    tol=1e-3, # Tolerância para critério de parada, se a perda for menor que isso, o treinamento para
    random_state=42 # Para garantir a reprodutibilidade dos resultados
)

#treino

modelo.fit(x_train, y_train) # Treina o modelo de Perceptron usando os dados de treinamento (x_train e y_train)
y_pred = modelo.predict(x_test) # Faz previsões usando o modelo treinado com os dados de teste (x_test) e armazena as previsões em y_pred
acuracia = accuracy_score(y_test, y_pred) # Calcula a acurácia do modelo comparando as previsões (y_pred) com os rótulos reais (y_test) usando a função accuracy_score e armazena o resultado em acuracia

print(f'Acurácia do modelo de Perceptron: {acuracia:.2f}') # Imprime a acurácia do modelo de Perceptron formatada com duas casas decimais.