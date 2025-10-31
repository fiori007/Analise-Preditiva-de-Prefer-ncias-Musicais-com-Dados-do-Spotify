import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_eval(items_df, listens_df):
    # por enquanto não o usamos, mas vamos garantir que seja referenciado:
    _ = listens_df.shape  # ou qualquer acesso simples
    X = items_df[["popularity", "duration_ms", "explicit"]].fillna(0)

    """
    Treina e avalia um modelo usando items_df e listens_df.
    items_df : DataFrame com colunas como ‘popularity’, ‘duration_ms’, ‘explicit’, ‘preferred’
    listens_df : DataFrame com histórico de escutas (por enquanto pode não ser usado no treino)
    """
    # Exemplo: usando apenas colunas de metadados
    X = items_df[["popularity", "duration_ms", "explicit"]].fillna(0)
    y = items_df["preferred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf
