import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import preprocess
from sklearn import (pipeline, base, preprocessing, ensemble,feature_selection,
                     model_selection,compose, impute, metrics)
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('Sinistros')
mlflow.sklearn.autolog()

## Dados
print("Carregando dados...")
sinistro = pd.read_csv('../data_sample/sample.csv')

df = sinistro.copy()
print(f'Foram carregados {sinistro.shape[0]} linhas e {sinistro.shape[1]} colunas.')

print("Pré-processando dados...")
preprocess.remove_columns(df)
preprocess.create_target(df)

# Split

target = 'acidente_grave'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.2, 
                                                                    random_state=42,
                                                                    stratify=y)

# Treinamento
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
        ('num_transformer', num_pipeline, compose.make_column_selector(dtype_include=np.number)),
        ('cat_transformer', cat_pipeline, compose.make_column_selector(dtype_include=[object])),
    ]
)

selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif, k=20)

model = ensemble.RandomForestClassifier(random_state=42,
                                        class_weight='balanced',
                                        max_depth=29,
                                        min_samples_leaf=10,
                                        min_samples_split=14,
                                        n_estimators=300,
                                        n_jobs=-1)

pipe = pipeline.Pipeline([
    ('data_types', preprocess.data_types()),
    ('features', preprocess.new_features()),
    ('preprocessor',preprocessor),
    ('select_features', selector),
    ('rnd_forest', model),
])

print("Iniciando treinamento...")
with mlflow.start_run(run_name=model.__str__()):
    
    pipe.fit(X_train, y_train)
    print("Modelo treinado!")
    print("Calculando métricas no treino (cross-validation)...")
    y_train_pred = model_selection.cross_val_predict(pipe, X_train, y_train, cv=3)
    y_train_proba = model_selection.cross_val_predict(pipe, X_train, y_train, cv=3, method='predict_proba')[:,1]
    
    precision = metrics.precision_score(y_train, y_train_pred)
    recall = metrics.recall_score(y_train, y_train_pred)
    f1_score = metrics.f1_score(y_train, y_train_pred)
    auc_pr = metrics.average_precision_score(y_train, y_train_proba)

    print('Precision_train: ', precision)
    print('Recall_train: ', recall)
    print('f1_score_train: ', f1_score)
    print('auc_pr_train: ', auc_pr)


    ## achando o threshold
    print("Encontrando melhor threshold...")
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_train,y_train_proba)
    min_recall = 0.9
    mask = recalls[:-1] >= min_recall
    index_best = np.argmax(precisions[:-1][mask])
    best_threshold = thresholds[mask][index_best]
    print(f"Threshold de {best_threshold} encontrado")

    print("Avaliando no conjunto de teste...")
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:,1]
    y_pred_threshold= ((y_pred_proba > best_threshold).astype(int)) ## predição com threshold

    precision_test = metrics.precision_score(y_test, y_pred_threshold)
    recall_test = metrics.recall_score(y_test, y_pred_threshold)
    f1_test = metrics.f1_score(y_test, y_pred_threshold)
    auc_pr_test = metrics.average_precision_score(y_test, y_pred_proba)
    log_loss_test = metrics.log_loss(y_test, y_pred)

    print('Precision_test: ', precision_test)
    print('Recall_test: ', recall_test)
    print('f1_score_test: ', f1_test)
    print('auc_pr_test: ', auc_pr_test)
    print('Log Loss: ', log_loss_test)

    mlflow.log_metrics({
        'train_Precision':precision,
        'train_Recall':recall,
        'train_F1':f1_score,
        'train_auc_pr':auc_pr,

        # with threshold to precision=0.9
        'test_precision':precision_test,
        'test_recall':recall_test,
        'test_auc_pr':auc_pr_test,
        'log_loss':log_loss_test,
        'test_f1':f1_test
    })

