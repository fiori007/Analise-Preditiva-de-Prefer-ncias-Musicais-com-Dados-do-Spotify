import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path  # Importa a biblioteca para lidar com caminhos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# -------------------------------------------------------------------
# 1. CARREGAMENTO DE DADOS (Dos seus arquivos .csv)
# -------------------------------------------------------------------

# Define os caminhos dos arquivos usando pathlib
# Isso garante compatibilidade entre Windows (com '\') e Linux/Mac (com '/')
base_path = Path("data") / "raw"
tracks_file = base_path / "tracks.csv"
artists_file = base_path / "artists.csv" # Corrigido do 'arti.csv' da primeira prompt
listens_file = base_path / "listens.csv"

# Tenta carregar os arquivos
try:
    df_tracks = pd.read_csv(tracks_file)
    df_artists = pd.read_csv(artists_file)
    df_listens = pd.read_csv(listens_file)
    print("Arquivos CSV carregados com sucesso de 'data/raw/'.")
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado.")
    print(f"Detalhe: {e}")
    print("\nPor favor, verifique se os arquivos estão nos caminhos corretos:")
    print(f" - {tracks_file.resolve()}")
    print(f" - {artists_file.resolve()}")
    print(f" - {listens_file.resolve()}")
    print("\nCertifique-se de executar este script a partir do diretório raiz do seu projeto TCC.")
    # Encerra o script se os arquivos não forem encontrados
    exit()
except Exception as e:
    print(f"Ocorreu um erro inesperado ao ler os arquivos: {e}")
    exit()

# -------------------------------------------------------------------
# 2. DEFINIÇÃO DA PREFERÊNCIA (Obj. 2) E ENGENHARIA DE FEATURES (Obj. 4)
# -------------------------------------------------------------------

print("Iniciando pré-processamento e engenharia de features...")

# Processa df_listens para features temporais e contagem
df_listens['played_at'] = pd.to_datetime(df_listens['played_at'])
df_listens['hour_of_day'] = df_listens['played_at'].dt.hour
df_listens['day_of_week'] = df_listens['played_at'].dt.weekday

# Agrega por usuário e música
df_user_track_agg = df_listens.groupby(['user_id', 'track_id']).agg(
    listen_count=('track_id', 'size'),
    avg_hour=('hour_of_day', 'mean'),
    avg_day=('day_of_week', 'mean')
).reset_index()

# **Objetivo 2: Operacionalização da Preferência**
# Definimos "preferência" = 1 se o usuário ouviu a música mais de uma vez.
df_user_track_agg['preference'] = (df_user_track_agg['listen_count'] > 1).astype(int)

# Processa df_tracks para features
df_tracks['explicit'] = df_tracks['explicit'].astype(int)
df_tracks['album_release_date'] = pd.to_datetime(df_tracks['album_release_date'], errors='coerce')
df_tracks['release_year'] = df_tracks['album_release_date'].dt.year

# Extrai o *primeiro* artista (artista principal) para o merge
# Remove aspas que podem estar presentes em listas de IDs
df_tracks['primary_artist_id'] = df_tracks['artist_ids'].apply(
    lambda x: str(x).split(',')[0].strip('[]"\' ')
)

# Seleciona colunas relevantes
# Adiciona .copy() para evitar o SettingWithCopyWarning
df_tracks_feats = df_tracks[['track_id', 'duration_ms', 'popularity', 'explicit', 'release_year', 'primary_artist_id']].copy() # <-- ALTERAÇÃO
df_artists_feats = df_artists[['artist_id', 'followers', 'popularity_artist']].copy() # <-- ALTERAÇÃO

# Converte IDs de artistas para string para garantir um merge consistente
df_artists_feats['artist_id'] = df_artists_feats['artist_id'].astype(str)
df_tracks_feats['primary_artist_id'] = df_tracks_feats['primary_artist_id'].astype(str)


# -------------------------------------------------------------------
# 3. FUSÃO DOS DADOS (Early Fusion - Obj. 6)
# -------------------------------------------------------------------

print("Mesclando DataFrames...")

# Junta os dados de escuta (com a target 'preference') com as features das músicas
data = pd.merge(df_user_track_agg, df_tracks_feats, on='track_id', how='left')

# Junta com as features dos artistas
data = pd.merge(data, df_artists_feats, left_on='primary_artist_id', right_on='artist_id', how='left')

print(f"Dados mesclados. Total de amostras: {len(data)}")

# -------------------------------------------------------------------
# 4. PRÉ-PROCESSAMENTO E DIVISÃO (Train/Test)
# -------------------------------------------------------------------

# Define colunas de features e a target
feature_cols = [
    'avg_hour', 'avg_day', # Features de comportamento
    # 'listen_count', <-- REMOVIDO! Esta era a causa do vazamento de dados.
    'duration_ms', 'popularity', 'explicit', 'release_year', # Features da música
    'followers', 'popularity_artist' # Features do artista
]
target_col = 'preference'

# Vamos passar o DataFrame 'data' diretamente para X e y.
data_clean = data.copy() 

# Verificamos NaNs SÓ na target.
data_clean = data_clean.dropna(subset=[target_col])

if data_clean.empty:
    print("ERRO: O DataFrame está vazio.")
    exit()
else:
    print(f"Amostras (pré-imputação): {len(data_clean)}") 
    
    # Verifica se há variação suficiente na target
    if data_clean[target_col].nunique() < 2:
        print(f"ERRO: A coluna target '{target_col}' só tem um valor (provavelmente só 0 ou só 1).")
        print("O modelo não pode ser treinado (precisa de exemplos das duas classes).")
        exit()
    else:
        X = data_clean[feature_cols]
        y = data_clean[target_col]

        # Divide os dados
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError as e: 
            print(f"ERRO ao dividir os dados: {e}")
            print(f"Total de amostras: {len(y)}. Contagem de classes (0 e 1): \n{y.value_counts()}")
            exit()


        # Cria um pipeline de pré-processamento
        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Aplica o pré-processamento
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        print(f"Dados divididos e processados: {len(X_train_processed)} treino, {len(X_test_processed)} teste.")

        # -------------------------------------------------------------------
        # 5. TREINAMENTO E AVALIAÇÃO (Obj. 5 & 8)
        # -------------------------------------------------------------------

        # Define os modelos
        # --- ALTERAÇÃO: Adicionando hiperparâmetros para regularização (combater overfitting) ---
        # Devido ao dataset muito pequeno (n=43), forçamos os modelos a serem mais simples.
        models = {
            "Random_Forest": RandomForestClassifier(
                random_state=42, 
                class_weight='balanced',
                n_estimators=50,  # Menos árvores
                max_depth=4        # Árvores bem curtas
            ),
            "Gradient_Boosting": GradientBoostingClassifier(
                random_state=42,
                n_estimators=50,  # Menos árvores
                max_depth=3        # Árvores bem curtas
            ),
            "XGBoost": XGBClassifier(
                random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss',
                n_estimators=50,  # Menos árvores
                max_depth=3        # Árvores bem curtas
            )
        }
        
        # Ajuste para classes desbalanceadas (comum em "preferências")
        # O XGBoost pode precisar de 'scale_pos_weight' se os dados forem muito desbalanceados
        # por enquanto, 'class_weight=balanced' no Random Forest já ajuda.

        # Variáveis para salvar o melhor modelo (baseado no F1-Score)
        best_f1 = -1.0
        best_model_name = ""
        best_model_object = None
        best_confusion_matrix = None

        # String para armazenar todos os relatórios
        metrics_report_str = "RELATÓRIO DE MÉTRICAS DOS MODELOS\n"
        metrics_report_str += "=" * 40 + "\n\n"

        for name, model in models.items():
            print(f"--- Treinando {name} ---")
            
            # Treina o modelo
            model.fit(X_train_processed, y_train)
            
            # Faz predições
            y_pred = model.predict(X_test_processed)
            y_prob = model.predict_proba(X_test_processed)[:, 1]

            # Calcula métricas (Obj 8)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                # average='macro' é útil para ver performance em ambas as classes igualmente
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                roc_auc = roc_auc_score(y_test, y_prob)
            except ValueError as e_roc:
                print(f"Aviso: Não foi possível calcular ROC-AUC para {name}. Erro: {e_roc}")
                roc_auc = np.nan
                f1_macro = f1 # fallback
            
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0, 
                                           target_names=['Não Preferência (0)', 'Preferência (1)'])

            # Adiciona ao relatório
            metrics_report_str += f"Modelo: {name}\n"
            metrics_report_str += "-" * 30 + "\n"
            metrics_report_str += f"Acurácia: {accuracy:.4f}\n"
            metrics_report_str += f"F1-Score (Weighted): {f1:.4f}\n"
            metrics_report_str += f"F1-Score (Macro): {f1_macro:.4f}\n"
            metrics_report_str += f"Precisão (Weighted): {precision:.4f}\n"
            metrics_report_str += f"Recall (Weighted): {recall:.4f}\n"
            metrics_report_str += f"ROC-AUC: {roc_auc:.4f}\n\n"
            metrics_report_str += "Classification Report:\n"
            metrics_report_str += report + "\n\n"
            metrics_report_str += "Matriz de Confusão:\n"
            metrics_report_str += str(cm) + "\n"
            metrics_report_str += "=" * 40 + "\n\n"
            
            # Usa o F1-Macro para escolher o melhor modelo,
            # pois é melhor para dados desbalanceados
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_model_name = name
                best_model_object = model
                best_confusion_matrix = cm

        print("Treinamento concluído.")

        # -------------------------------------------------------------------
        # 6. SALVANDO OS RESULTADOS (Obj. 8)
        # -------------------------------------------------------------------

        # 1. Salvar o arquivo .txt com todas as métricas
        try:
            with open("model_metrics.txt", "w", encoding="utf-8") as f:
                f.write(metrics_report_str)
            print("Arquivo 'model_metrics.txt' salvo com sucesso.")
        except Exception as e:
            print(f"Erro ao salvar arquivo de métricas: {e}")

        # 2. Salvar o melhor modelo (objeto)
        if best_model_object:
            model_filename = f"{best_model_name}_best_model.joblib"
            try:
                # Salva o pipeline completo (preprocessador + modelo)
                # Embora aqui estejamos salvando só o modelo,
                # o ideal seria salvar um pipeline
                joblib.dump(best_model_object, model_filename)
                
                # Salva também o pré-processador!
                joblib.dump(preprocessor, "preprocessor.joblib")
                print(f"Melhor modelo ('{best_model_name}') salvo como '{model_filename}'.")
                print("Pré-processador salvo como 'preprocessor.joblib'.")
            except Exception as e:
                print(f"Erro ao salvar modelo: {e}")

        # 3. Salvar a imagem da melhor Matriz de Confusão
        if best_confusion_matrix is not None:
            try:
                plt.figure(figsize=(10, 7))
                sns.heatmap(best_confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Não Preferência (0)', 'Preferência (1)'], 
                            yticklabels=['Não Preferência (0)', 'Preferência (1)'])
                plt.title(f'Matriz de Confusão - {best_model_name} (Melhor F1-Macro: {best_f1:.4f})')
                plt.xlabel('Predito')
                plt.ylabel('Verdadeiro')
                plt.savefig("best_confusion_matrix.png")
                print("Imagem 'best_confusion_matrix.png' salva com sucesso.")
                # plt.show() # Descomente se quiser ver o gráfico
            except Exception as e:
                print(f"Erro ao salvar imagem da matriz de confuão: {e}")