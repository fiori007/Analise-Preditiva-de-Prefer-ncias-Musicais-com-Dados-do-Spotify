import pandas as pd

DEFAULT_PLAYCOUNT_THRESHOLD = 2

def build_preference_labels(items_df: pd.DataFrame,
                            listens_df: pd.DataFrame,
                            playcount_threshold=DEFAULT_PLAYCOUNT_THRESHOLD):
    counts = listens_df.groupby("track_id").size().rename("playcount").reset_index()
    merged = items_df.merge(counts, on="track_id", how="left").fillna({"playcount": 0})
    merged["preferred"] = (merged["playcount"] >= playcount_threshold).astype(int)
    return merged

def add_temporal_features(listens_df: pd.DataFrame):
    out = listens_df.copy()
    out["hour"] = out["played_at"].dt.hour
    out["dow"] = out["played_at"].dt.dayofweek
    return out

def user_profile_features(items_with_label: pd.DataFrame):
    # Como não temos mais audio_features, vamos usar colunas disponíveis
    # Por exemplo: popularidade e duração
    liked = items_with_label[items_with_label["preferred"] == 1]
    if liked.empty:
        return {}
    # Exemplo simples: média de popularidade
    profile = {
        "avg_popularity": liked["popularity"].mean(),
        "avg_duration_ms": liked["duration_ms"].mean()
    }
    return profile
