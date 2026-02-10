#%%
## Imports 
import pandas as pd
import numpy as np
from sklearn import base

def remove_columns(df):
    to_remove = ['id_sinistro', 'tipo_acidente_primario',
                'ano_mes_sinistro', 'ano_sinistro',
                'logradouro', 'numero_logradouro',
                'municipio', 'regiao_administrativa',
                'administracao', 'conservacao',
                'jurisdicao', 'latitude',
                'longitude', 'tipo_registro']
    df.drop(columns=to_remove, inplace=True)
    df.drop_duplicates(inplace=True)
    return df.reset_index(drop=True)

## juntar acidentes fatal e grave em uma coluna
def create_target(df):
    df['acidente_grave'] = ((df['gravidade_fatal'] > 0) | (df['gravidade_grave'] > 0)).astype(int)
    return df

def create_periodos(time):
            if 6 <= time < 12:
                return 'manha'
            elif 12 <= time < 18:
                return 'tarde'
            elif 18 <= time < 24:
                return 'noite'
            elif 0 <= time < 6:
                return 'madrugada'
            else:
                return 'nao_disponivel'

class data_types(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        veiculos = [col for col in X.columns if col.startswith('tp_veiculo_')]
        gravidade = [col for col in X.columns if col.startswith('gravidade_')]
        tipos = [col for col in X.columns if col.startswith('tp_sinistro_')]
        ## 'hora_sinistro': object -> int
        X['hora_sinistro'] = pd.to_numeric(X['hora_sinistro'].str[:2], errors='coerce')
        X['hora_sinistro'] = X['hora_sinistro'].fillna(-1).astype(int)
        ## 'tp_sinistro_': 'S' -> 1
        for col in tipos:
            X[col] = X[col].replace('S','1')
            X[col] = pd.to_numeric(X[col], errors='coerce')
        ## 'data_sinistro': object -> datetime
        X['data_sinistro'] = pd.to_datetime(X['data_sinistro'], format='%d/%m/%Y')
        # preenchendo NaN com 0 nas colunas binárias
        X[veiculos] = X[veiculos].fillna(0)
        X[gravidade] = X[gravidade].fillna(0)
        X[tipos] = X[tipos].fillna(0)
        # transformando 'gravidade_fatal' em binário
        X['gravidade_fatal'] = (X['gravidade_fatal']>0).astype(int)
        X['gravidade_grave'] = (X['gravidade_grave']>0).astype(int)
        return X
    
class new_features(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, features):
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        ## Período do dia que ocorreu o acidente
        ## Temos a ideia de que acidentes na madrugada tendem a ser mais graves
        X['periodo_dia'] = X['hora_sinistro'].apply(create_periodos)
        ## Temos a ideia de que acidentes no fim de semana tendem a ser mais graves por estar asoociado a festas
        X['dia_semana'] = X['data_sinistro'].dt.day_name()
        ## fim de semana? sim ou não
        X['fim_de_semana'] = ((X['dia_semana'] == 'Friday') | 
                                    (X['dia_semana'] == 'Saturday') | 
                                    (X['dia_semana'] == 'Sunday')).astype(int)
        ## acidentes em maiores velocidades tendem ser mais graves
        X['alta_velocidade_provavel'] = (X['tipo_via'] == 'RODOVIAS').astype(int)
        ## 'claridade_escuro' para indicar se o acidente ocorreu durante a luz da manhã/tarde ou no escuro da noite/madrugada
        ## claridade menor pode aumentar a incidência de acidentes, principalmente se somado a altas velocidades ('provavel_velocidade_noturna)
        X['claridade_escuro'] = ((X['periodo_dia'] == 'madrugada') | (X['periodo_dia'] == 'noite')).astype(int)
        X['provavel_velocidade_noturna'] = ((X['claridade_escuro']>0) & (X['alta_velocidade_provavel']>0)).astype(int)
        ## 'qtd_pessoas' -> soma quantidade de pessoas envolvidas 
        ## 'qtd_veiculos' -> soma quantidade de veículos envolvidos
        veiculos = [col for col in X.columns if col.startswith('tp_veiculo_')]
        gravidade = [col for col in X.columns if col.startswith('gravidade_')]
        X['qtd_pessoas'] = X[gravidade].sum(axis=1).astype(int)
        X['qtd_veiculos'] = X[veiculos].sum(axis=1).astype(int)
        X['qtd_tipos_sinistro'] = (X[[c for c in X.columns if c.startswith('tp_sinistro_')]].sum(axis=1)).astype(int)
        ## tem caminhão ou ônibus?
        ## por ser veiculos pesados a fatalidade pode ser maior
        X['veiculo_pesado'] = ((X['tp_veiculo_caminhao']>0) | (X['tp_veiculo_onibus']>0)).astype(int)
        ## quantidade de pessoas por veículo
        X['pessoas_por_veiculos'] = (X['qtd_pessoas']/X['qtd_veiculos'])
        ## tratando pessoas por veiculos
        ## para 'qtd_veiculos'=0 ocasionará em valores infinitos
        X['pessoas_por_veiculos'] = X['pessoas_por_veiculos'].replace([np.inf,-np.inf],np.nan)
        ## Para valores não preenchidos pessoas por veiculos será igual a quantidade de pessoas, se não preenchido será 0
        X['pessoas_por_veiculos'] = X['pessoas_por_veiculos'].fillna(X['qtd_pessoas'])
        X['pessoas_por_veiculos'] = X['pessoas_por_veiculos'].fillna(0)
        
        to_drop= ['data_sinistro','hora_sinistro'] + gravidade
        X = X.drop(columns=to_drop)
        
        if self.features:
            return X[self.features]
        return X

