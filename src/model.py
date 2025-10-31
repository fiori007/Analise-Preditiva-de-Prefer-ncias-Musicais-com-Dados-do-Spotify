import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

NUMERIC = [
    "danceability", "energy", "acousticness", "instrumentalness", "liveness",
    "speechiness", "valence", "tempo", "loudness", "popularity",
    "followers", "popularity_artist"
]
CATEG = ["mode", "key", "time_signature"]
TARGET = "preferred"

def train_and_eval(items_path="data/processed/items_labeled.csv"):
    """
    Treina e avalia um modelo usando o arquivo items_labeled.csv.
    items_path : caminho para o CSV com os itens rotulados.
    """
    df = pd.read_csv(items_path)

    # Garante que as colunas existam
    for c in NUMERIC:
        if c not in df.columns:
            df[c] = 0
    for c in CATEG:
        if c not in df.columns:
            df[c] = "UNK"

    df = df.fillna({c: 0 for c in NUMERIC}).fillna({c: "UNK" for c in CATEG})

    X = df[NUMERIC + CATEG]
    y = df[TARGET]

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEG)
    ])

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # Verifica se todas as classes têm pelo menos 2 amostras antes da estratificação
    if len(y.unique()) > 1 and all(y.value_counts() >= 2):
        stratify_opt = y
    else:
        stratify_opt = None
        print("⚠️ Aviso: classes insuficientes para estratificação (alguma tem <2 amostras).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify_opt, random_state=42
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            proba = None

    print(classification_report(y_test, preds, digits=4))
    if proba is not None:
        try:
            print("ROC-AUC:", roc_auc_score(y_test, proba))
        except Exception as e:
            print("ROC-AUC indisponível:", e)
    else:
        print("ROC-AUC não calculado pois predict_proba indisponível.")

    return pipe
