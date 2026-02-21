#%% 
# Imports
import pandas as pd
import numpy as np

from sklearn import(model_selection, impute , preprocessing, compose, metrics,
                    ensemble, pipeline, dummy, tree, feature_selection)
import mlflow


## configs
pd.set_option('display.max_columns', None)

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Sinistro")

#%% iniciando log de resultados dos modelos
mlflow.set_tracking_uri('http://localhost:5000')

mlflow.set_experiment("Sinistro")
#%%
# Import
## fontes dos dados: https://dadosabertos.sp.gov.br/dataset/sinistros-infosiga
raw = pd.read_csv('../data/sinistros_2022-2025.csv', 
                  encoding='latin-1', 
                  on_bad_lines='skip', 
                  sep=';', 
                  decimal=',')
sinistro = raw.copy()

#%%
sample = sinistro.head(250)


#%%
sinistro.shape


# %%
sinistro.dtypes


#%%
sinistro.head(5)
## -colunas de gravidade indicam quantidade de pessoas em tal situação,
## -não são binárias


#%%
sinistro.info()


#%%
## removendo linhas duplicadas e colunas não desejadas
to_remove = ['id_sinistro', 'tipo_acidente_primario',
             'ano_mes_sinistro', 'ano_sinistro',
             'logradouro', 'numero_logradouro',
             'municipio', 'regiao_administrativa',
             'administracao', 'conservacao',
             'jurisdicao', 'latitude',
             'longitude', 'tipo_registro']
sinistro.drop(columns=to_remove, inplace=True)
sinistro.drop_duplicates(inplace=True)


#%% 
# data types
## hora_sinistro (object -> int)
sinistro.dtypes
sinistro['hora_sinistro'] = pd.to_numeric(sinistro['hora_sinistro'].str[:2], errors='coerce')

## tp_sinistro_ 'S' -> 1
tipos = [col for col in sinistro.columns if col.startswith('tp_sinistro_')]
for col in tipos:
    sinistro[col] = sinistro[col].replace('S','1')
    sinistro[col] = pd.to_numeric(sinistro[col], errors='coerce')


#%%
## criar coluna 'periodo_dia'
def criar_periodos(hora):
    if 6 <= hora < 12:
        return 'manha'
    elif 12 <= hora < 18:
        return 'tarde'
    elif 18 <= hora < 24:
        return 'noite'
    elif hora < 6:
        return 'madrugada'
    else:
        return 'nao_disponivel'

sinistro['periodo_dia'] = sinistro['hora_sinistro'].apply(criar_periodos)
sinistro.drop(columns='hora_sinistro',inplace=True)

## 'data_sinistro' = object
## object -> datatime
sinistro['data_sinistro'] = pd.to_datetime(sinistro['data_sinistro'], format='%d/%m/%Y')

## criar coluna 'dia_semana'
sinistro['dia_semana'] = sinistro['data_sinistro'].dt.day_name()

sinistro.drop(columns='data_sinistro',inplace=True)


#%% 
# preenchendo NaN com 0 nas colunas binárias
veiculos = [col for col in sinistro.columns if col.startswith('tp_veiculo_')]
gravidade = [col for col in sinistro.columns if col.startswith('gravidade_')]

sinistro[veiculos] = sinistro[veiculos].fillna(0)
sinistro[gravidade] = sinistro[gravidade].fillna(0)
sinistro[tipos] = sinistro[tipos].fillna(0)


#%% 
# criando coluna 'qtd_pessoas' e 'qtd_veiculos'
sinistro['qtd_pessoas'] = sinistro[gravidade].sum(axis=1).astype(int)
sinistro['qtd_veiculos'] = sinistro[veiculos].sum(axis=1).astype(int)
sinistro['qtd_tipos_sinistro'] = (sinistro[[c for c in sinistro.columns if c.startswith('tp_sinistro_')]].sum(axis=1)).astype(int)


#%% 
# transformando 'gravidade_fatal' em binário
sinistro['gravidade_fatal'] = (sinistro['gravidade_fatal']>0).astype(int)
sinistro['gravidade_grave'] = (sinistro['gravidade_grave']>0).astype(int)


#%% 
# juntando coluna 'gravidade_fatal' com 'gravidade_grave'
sinistro['acidente_grave'] = ((sinistro['gravidade_fatal'] > 0) | (sinistro['gravidade_grave'] > 0)).astype(int)
sinistro.drop(columns=gravidade, inplace=True)


#%%
# reset_index
sinistro.reset_index(inplace=True, drop=True)


#%% 
# Criando colunas novas
sinistro['veiculo_pesado'] = ((sinistro['tp_veiculo_caminhao']>0) | (sinistro['tp_veiculo_onibus']>0)).astype(int)
sinistro['pessoas_por_veiculos'] = (sinistro['qtd_pessoas']/sinistro['qtd_veiculos'])
sinistro['fim_de_semana'] = ((sinistro['dia_semana'] == 'Friday') | 
                             (sinistro['dia_semana'] == 'Saturday') | 
                             (sinistro['dia_semana'] == 'Sunday')).astype(int)
sinistro['alta_velocidade_provavel'] = (sinistro['tipo_via'] == 'RODOVIAS').astype(int)
sinistro['claridade_escuro'] = ((sinistro['periodo_dia'] == 'madrugada') | (sinistro['periodo_dia'] == 'noite')).astype(int)
sinistro['provavel_velocidade_noturna'] = ((sinistro['claridade_escuro']>0) & (sinistro['alta_velocidade_provavel']>0)).astype(int)


#%% 
# tratando pessoas por veiculos
sinistro['pessoas_por_veiculos'] = sinistro['pessoas_por_veiculos'].replace([np.inf,-np.inf],np.nan)
sinistro['pessoas_por_veiculos'] = sinistro['pessoas_por_veiculos'].fillna(sinistro['qtd_pessoas'])
sinistro['pessoas_por_veiculos'] = sinistro['pessoas_por_veiculos'].fillna(0)


#%% 
# Split
target = 'acidente_grave'
X = sinistro.drop(columns=[target])
y = sinistro[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.2, 
                                                                    random_state=42,
                                                                    stratify=y)

#%%
# Treinamento
## nomes das colunas numéricas e categóricas
num_cols = X_train.select_dtypes(include=['int64','float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

#%% 
# Pipelines
num_pipeline = pipeline.Pipeline([
    ('scaler', preprocessing.RobustScaler()),
])

cat_pipeline = pipeline.Pipeline([
    ('input_cat',impute.SimpleImputer(strategy='most_frequent')),
    ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = compose.ColumnTransformer(
    transformers=[
        ('num_transformer', num_pipeline, num_cols),
        ('cat_transformer', cat_pipeline, cat_cols),
    ]
)

select_features = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif, k=17)

model = ensemble.RandomForestClassifier(random_state=42,
                                        class_weight='balanced',
                                        max_depth=15,
                                        min_samples_split=15,
                                        min_samples_leaf=2,
                                        n_estimators=100,
                                        n_jobs=1,)
# {'rnd_forest__max_depth': 14, 'rnd_forest__min_samples_leaf': 2, 'rnd_forest__min_samples_split': 6, 'rnd_forest__n_estimators': 408}
# {'rnd_forest__max_depth': 12, 'rnd_forest__min_samples_leaf': 2, 'rnd_forest__min_samples_split': 16, 'rnd_forest__n_estimators': 125}
# {'rnd_forest__max_depth': 15, 'rnd_forest__min_samples_leaf': 2, 'rnd_forest__min_samples_split': 15, 'rnd_forest__n_estimators': 300}
pipe = pipeline.Pipeline([
    ('preprocessor',preprocessor),
    ('best_features', select_features),
    ('rnd_forest', model),
])


#%% 
# Achando as melhores features para simplificar o modelo
# best_model = pipe.named_steps['rnd_forest']
# features = pipe.named_steps['preprocessor'].get_feature_names_out()
# importances = best_model.feature_importances_
# features_importances = pd.DataFrame({'feature':features,
#                                      'importance':importances}).sort_values(by='importance', 
#                                                                             ascending=False)
# best_features = features_importances['feature'].head(20).values


#%% 
# # Extraindo o nome das features
# best_features = [col.replace('num_transformer__','').replace('cat_transformer__','') for col in best_features]
# best_features_names = [col for col in best_features if col in X_train.columns]

# best_num_cols = X_train[best_features_names].select_dtypes(include=['int64','float64']).columns.tolist()
# best_cat_cols = X_train[best_features_names].select_dtypes(include=['object']).columns.tolist()


#%% 
# Melhores hiperparâmetros
from scipy.stats import randint

## Primeiro, rodar alguns random search para achar um bom range de valores para colocar no grid search

## para RandomSearchCV
param_distributions = {'rnd_forest__max_depth' : randint(low=10, high=50),
                 'rnd_forest__min_samples_split': randint(low=5,high=20),
                 'rnd_forest__n_estimators': randint(low=50,high=500),
                 'rnd_forest__min_samples_leaf': [2],
                 }
## remover hiperparâmetros no model
## mudar param_grid para param_distributions no grid_search
## adicionar n_iter=10

## para GridSearchCV
param_grid = {'rnd_forest__max_depth' : [15,25,50],
                 'rnd_forest__min_samples_split': [5,10,15],
                 'rnd_forest__n_estimators': [150,300],
                 'rnd_forest__min_samples_leaf':[2],
                 }
## remover n_iter
## param_distributions para param_grid
## remover os hiperparâmetros no model

grid_search = model_selection.GridSearchCV(pipe,
                                           param_grid=param_grid,
                                           cv=3,
                                           scoring='f1',)

grid_search.fit(X_train, y_train)

print('Best params: ',grid_search.best_params_)


#%% Dummy classifier
dummy_clf = pipeline.make_pipeline(
    preprocessor,
    dummy.DummyClassifier()
)
dummy_clf.fit(X_train, y_train)
most_frequent = any(dummy_clf.predict(X_train))
print(f'Most frequent class: {most_frequent}')
if most_frequent == False:
    most_frequent = 0
else:
    most_frequent = 1
print(f'Se o modelo chutar {most_frequent} para todas as predições, ele está certo em {(model_selection.cross_val_score(dummy_clf, X_train, y_train, cv=5, scoring='accuracy').mean())}% das vezes')


#%% 
# testando modelo com best_features_names e logando os resultados

with mlflow.start_run(run_name=model.__str__()):
    
    pipe.fit(X_train, y_train)
    y_train_pred_best = model_selection.cross_val_predict(pipe, X_train, y_train, cv=3)
    y_train_proba_best = model_selection.cross_val_predict(pipe,X_train, y_train, cv=3, method='predict_proba')

    y_test_pred_best = pipe.predict(X_test)
    y_test_proba_best = pipe.predict_proba(X_test)

    matrix_best = metrics.confusion_matrix(y_train, y_train_pred_best)
    precision_best = metrics.precision_score(y_train, y_train_pred_best)
    recall_best = metrics.recall_score(y_train, y_train_pred_best)
    f1_score_best = metrics.f1_score(y_train, y_train_pred_best)
    auc_pr_best = metrics.average_precision_score(y_train, y_train_proba_best[:,1])

    test_precision_best = metrics.precision_score(y_test, y_test_pred_best)
    test_recall_best = metrics.recall_score(y_test, y_test_pred_best)
    test_auc_pr_best = metrics.average_precision_score(y_test, y_test_proba_best[:,1])

    print('Precision ',precision_best)
    print('Recall ',recall_best)
    print('F1 ',f1_score_best)
    print('AUC-PR ', auc_pr_best)
    print(matrix_best)

    mlflow.log_metrics({
        'train_Precision':precision_best,
        'train_Recall':recall_best,
        'train_F1':f1_score_best,
        'train_auc_pr':auc_pr_best,
        'test_auc_pr': test_auc_pr_best,
        'test_recall': test_recall_best,
        'test_precision': test_precision_best,
    })


#%%
top_best_features = pd.DataFrame({
    'feature': pipe.named_steps['preprocessor'].get_feature_names_out(),
    'importance':pipe.named_steps['best_features'].scores_
})
top_best_features.sort_values(by='importance', ascending=False).head(20).reset_index(drop=True)


#%% 
# Achando melhor threshold
y_probas_best = model_selection.cross_val_predict(pipe, X_train, y_train, cv=3, method='predict_proba')[:,1]
precisions, recalls, thresholds = metrics.precision_recall_curve(y_train, y_probas_best)

min_recall = 0.9
mask = recalls[:-1] >= min_recall

best_index = np.argmax(precisions[:-1][mask])
best_threshold = thresholds[mask][best_index]


#%%
w_threshold = (y_test_proba_best[:,1] >= best_threshold).astype(int)
print(metrics.classification_report(y_test, w_threshold))


#%%
class predictor:
    def __init__(self, pipeline, threshold):
        self.pipeline = pipeline
        self.threshold = threshold
    
    def  predict(self, X):
        data = X.copy()
        proba = self.pipeline.predict_proba(data)[:,1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        data = X.copy()
        return self.pipeline.predict_proba(data)
    
final_model = predictor(
    pipeline=pipe,
    threshold = best_threshold
)
