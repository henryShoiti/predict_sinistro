import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn import (pipeline, base, preprocessing, ensemble, dummy, 
                     model_selection,compose, impute, metrics)
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('Sinistros')
mlflow.sklearn.autolog()
import preprocess

## Dados
sinistro = pd.read_csv('../data/sinistros_2022-2025.csv', 
                       encoding='latin-1', 
                       on_bad_lines='skip', 
                       sep=';', 
                       decimal=',')
df = sinistro.copy()

preprocess.remove_columns(df)
preprocess.create_target(df)

best_features = [
    'tipo_via',
    'qtd_pessoas', 
    'qtd_veiculos', 
    'qtd_tipos_sinistro', 
    'tp_sinistro_outros', 
    'pessoas_por_veiculos', 
    'tp_sinistro_tombamento', 
    'provavel_velocidade_noturna', 
    'tp_sinistro_choque', 
    'tp_veiculo_automovel', 
    'tp_veiculo_nao_disponivel', 
    'tp_veiculo_motocicleta', 
    'tp_sinistro_colisao_frontal', 
    'claridade_escuro', 
    'tp_veiculo_caminhao'
]
best_num_features = best_features[1:]
best_cat_features = ['tipo_via']

# Split
target = 'acidente_grave'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.2, 
                                                                    random_state=42,
                                                                    stratify=y)

# Treinamento
## nomes das colunas numéricas e categóricas
num_cols = X_train.select_dtypes(include=['int64','float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

## Pipelines
num_pipeline = pipeline.Pipeline([
    ('scaler', preprocessing.RobustScaler()),
])

cat_pipeline = pipeline.Pipeline([
    ('input_cat',impute.SimpleImputer(strategy='most_frequent')),
    ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = compose.ColumnTransformer(
    transformers=[
        ('num_transformer', num_pipeline, best_num_features),
        ('cat_transformer', cat_pipeline, best_cat_features),
    ]
)

model = ensemble.RandomForestClassifier(random_state=42,
                                        class_weight='balanced',
                                        max_depth=3,
                                        min_samples_split=6,
                                        n_estimators=6)

pipe = pipeline.Pipeline([
    ('data_types', preprocess.data_types()),
    ('features', preprocess.new_features(features=best_features)),
    ('preprocessor',preprocessor),
    ('rnd_forest', model)
])

with mlflow.start_run(run_name=model.__str__()):
    
    pipe.fit(X_train, y_train)
    y_train_pred = model_selection.cross_val_predict(pipe, X_train, y_train, cv=5)
    y_train_proba = model_selection.cross_val_predict(pipe,X_train, y_train, cv=5, method='predict_proba')
    
    accuracy = metrics.accuracy_score(y_train, y_train_pred)
    matrix = metrics.confusion_matrix(y_train, y_train_pred)
    precision = metrics.precision_score(y_train, y_train_pred)
    recall = metrics.recall_score(y_train, y_train_pred)
    f1_score = metrics.f1_score(y_train, y_train_pred)
    roc_auc = metrics.roc_auc_score(y_train, y_train_pred)
    auc_pr = metrics.average_precision_score(y_train, y_train_proba[:,1])

    precisions, recalls, thresholds = metrics.precision_recall_curve(y_train,y_train_proba[:,1])
    min_recall = 0.9
    mask = recalls[:-1] >= min_recall
    index_best = np.argmax(precisions[:-1][mask])
    best_threshold = thresholds[mask][index_best]

    y_test_proba = pipe.predict_proba(X_test)[:,1]
    y_test_pred = ((y_test_proba > best_threshold).astype(int)) ## predição com threshold

    precision_test = metrics.precision_score(y_test, y_test_pred)
    recall_test = metrics.recall_score(y_test, y_test_pred)
    auc_pr_test = metrics.average_precision_score(y_test, y_test_proba)

    mlflow.log_metrics({
        'train_Accuracy':accuracy,
        'train_Precision':precision,
        'train_Recall':recall,
        'train_F1':f1_score,
        'train_roc_auc':roc_auc,
        'train_auc_pr':auc_pr,

        'test_precision':precision_test,
        'test_recall':recall_test,
        'test_auc_pr':auc_pr_test
    })

