# Desafio IndustriAll
## Inicialização
Este código foi implementado utilizando Python v3.10.6, e as libs necessárias para rodar os scripts estão disponíveis no arquivo [requirements.txt](./requirements.txt). Para instalar as libs, basta rodar o comando no terminal:

~~~bash
$ pip install -r requirements.txt
~~~

## Rodando
Para rodar o script completo, basta rodar o arquivo [main.py](./main.py), que rodará os scripts de preprocessamento dos dados, treinamento dos modelos e teste dos modelos treinados. Caso queira rodar os arquivos separados, é necessário se atentar para os arquivos que são pré-requisitos deste.

### main.py
Arquivo com o script completo de treinamento e teste dos modelos, incluindo preprocessamento dos dados. Necessita que os dados estejam na pasta [data](./data/).

Ele roda os arquivos [data_preprocessing.py](./data_preprocessing.py), [data_learning.py](./data_learning.py) e [data_detection_test.py](./data_detection_test.py).

### data_preprocessing.py
Arquivo que roda o preprocessamento dos dados de entrada dos algoritmos. Ele realiza a leitura dos arquivos da pasta [data](./data/), trata valorez faltantes, realiza a normalização dos dados e separa os dados entre as bases de treinamento e teste. Ele gera como saída cinco arquivos:

- [data/data_base.csv](./data/data_base.csv): base de dados montada com todos os dados na pasta [data](./data/);
- [data/preprocessed_base.csv](./data/preprocessed_base.csv): base de dados preprocessados (com valores faltantes tratados), mas não normalizados;
- [data/data_normalized.pkl](./data/data_normalized.pkl): base de dados normalizada;
- [data_preprocessed.pkl](./data_preprocessed.pkl): base de dados separada entre dados de treino e teste; e
- [data_scaler.pkl](./data_scaler.pkl): normalizador a ser utilizado para preprocessar entradas a serem preditas pelos modelos.

### data_learning.py
Script que realiza o treinamento dos modelos de machine learning com os dados contidos no arquivo [data_preprocessed.pkl](./data_preprocessed.pkl). Precisa que o arquivo exista para rodar. Este script realiza o treinamento dos seguintes modelos:

- Decision Tree Classifier
- Naive Bayes Classifier
- K-Nearest Neighbors Classifier
- Logistic Regression Classifier
- Suport Vector Machine
- Neural Network Classifier

Ele gera o seguinte arquivo:

- [data_classifiers.pkl](./data_classifiers.pkl): arquivo com os modelos treinados na ordem descrita acima, para utilização posterior.

### data_detection_test.py
Script que testa o desempenho dos algoritmos em detectar as primeiras três ocorrências de anomalias na base de dados original. Ele depende dos arquivos [data/preprocessed_base.csv](./data/preprocessed_base.csv), [data_classifiers.pkl](./data_classifiers.pkl) e [data_scaler.pkl](./data_scaler.pkl). Ele salva seus resultados no arquivo [data_classifiers_test.pkl](./data_classifiers_test.pkl).

### data_visualization.py
Script que executa o plot dos dados dos sensores da base de dados original não normalizada. Depende do arquivo [data/preprocessed_base.csv](./data/preprocessed_base.csv).

### data_testing_results_visualization.py
Script que executa o plot dos dados de três testes do arquivo [data_detection_test.py](./data_detection_test.py):

- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Neural Network Classifier

Depende do arquivo [data_classifiers_test.pkl](./data_classifiers_test.pkl).