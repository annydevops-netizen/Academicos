# Fundamentos das Redes Neurais Artificiais e do Perceptron,  exercicio proposto -  unidade 1 : 
 
   *Para consolidar o aprendizado sobre o funcionamento do Perceptron e
   fortalecer sua compreensão sobre classificação binária,
   é proposto um desafio que vai além da teoria:
   buscar, selecionar e analisar uma base de dados real da plataforma Kaggle
   que trate de um problema binário. O objetivo é permitir que você vivencie 
   o processo completo que profissionais de Machine Learning enfrentam no mundo real, 
   desde encontrar um dataset relevante até formular um problema com clareza, 
   preparar os dados e aplicar um modelo 
   simples capaz de gerar indícios imediatos.
O Kaggle é uma plataforma rica em dados do mundo real, 
e vários de seus conjuntos tratam diretamente de decisões “sim ou não”,
“saudável ou não”, “fraude ou não”, “sobreviveu ou não”, 
tornando-se o ambiente perfeito para colocar em prática os conceitos estudados sobre Perceptron. 
A ideia do desafio é que você navegue pela plataforma e encontre um conjunto de dados
cuja tarefa principal seja a classificação binária,
como prever a sobrevivência no Titanic, detectar fraudes em cartões de crédito,
prever aprovação de crédito ou detectar ocupação em sensores inteligentes.
A escolha é livre, 
o importante é identificar um conjunto que desperte seu interesse
e ofereça variáveis que possam ser convertidas em entradas 
numéricas coerentes com o que discutimos teoricamente.*"


# Titanic Perceptron

Projeto de Machine Learning que utiliza um Perceptron para prever a sobrevivência dos passageiros do Titanic com base em características como Idade, Classe e Tarifa.

*A Machine Learning project using a Perceptron to predict Titanic passengers' survival based on features like Age, Class, and Fare.*

# 📂 Estrutura do Projeto / Project Structure

trabalhos_academicos/
│

  ├─ Data/

│   └─ titanic.csv

├─ titanic-perceptron.py


└─ README.md

# 💻 Tecnologias / Technologies

🐍 Python 3.x

📊 Pandas, NumPy

🤖 Scikit-learn

# ⚙️ Como Rodar / How to Run

*Criar e ativar um ambiente virtual (opcional, mas recomendado)*

Instalar dependências:

> pip install -r requirements.txt

Executar o script:

> python titanic-perceptron.py


#📈 Resultados / Results
Acurácia do Perceptron simples: ~67%
Mais features ou modelos avançados podem melhorar o desempenho

# ⚠️ Observações / Notes
Dataset utilizado: Titanic Dataset – Kaggle
O código é para aprendizado e experimentação; não está otimizado para produção
