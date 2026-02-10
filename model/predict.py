import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://localhost:5000')

list_models = mlflow.search_registered_models(filter_string="name = 'predict_sinistros'")
latest_version = max([i.version for i in list_models[0].latest_versions])

model = mlflow.sklearn.load_model(f'models:/predict_sinistros/{latest_version}')

new_data = pd.read_csv('../data/sinistros_2015-2021.csv', 
                       encoding='latin-1', 
                       on_bad_lines='skip', 
                       sep=';', 
                       decimal=',')

new_data = new_data.head(150)

new_predictions = model.predict(new_data)

new_data['predictions'] = new_predictions

print(new_data[['gravidade_grave','gravidade_fatal','predictions']])