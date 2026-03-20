import pandas as pd
import mlflow
import mlflow.sklearn
import sklearn.metrics as metrics
import preprocess

mlflow.set_tracking_uri('http://localhost:5000')

MODEL_NAME = 'new_model'

list_models = mlflow.search_registered_models(filter_string=f"name='{MODEL_NAME}'")
latest_version = max([i.version for i in list_models[0].latest_versions])

model = mlflow.sklearn.load_model(f'models:/{MODEL_NAME}/{latest_version}')

print('Carregando dados...')
new_data = pd.read_csv('../data_sample/new_data.csv')

print('Pré-processando os dados...')
target = 'acidente_grave'
preprocess.remove_columns(new_data)
preprocess.create_target(new_data)

X = new_data.drop(columns=target)
y = new_data[target]

print('Dados pré-processados com sucesso!')

print('Gerando predições...')
new_predictions = model.predict(X)
new_probas = model.predict_proba(X)[:,1]
print('Predições concluídas!')

new_data['predictions'] = new_predictions
new_data['probas'] = new_probas
print(new_data[['acidente_grave','predictions','probas']])

print(f'Precision score: {metrics.precision_score(y, new_predictions)}')
print(f'Recall score: {metrics.recall_score(y, new_predictions)}')