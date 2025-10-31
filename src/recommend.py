import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def recommend_content_based(items_df: pd.DataFrame, liked_track_ids: list, top_k=20):
    # Vamos usar colunas de metadados em vez de audio_features
    feature_cols = ["popularity", "duration_ms", "explicit"]
    df = items_df.dropna(subset=feature_cols).copy()
    if df.empty or not liked_track_ids:
        return pd.DataFrame(columns=["track_id", "track_name", "artist_names", "score_cb"])
    scaler = StandardScaler()
    M = scaler.fit_transform(df[feature_cols])
    id_to_idx = {tid: i for i, tid in enumerate(df["track_id"])}
    liked_idx = [id_to_idx[tid] for tid in liked_track_ids if tid in id_to_idx]
    if not liked_idx:
        return pd.DataFrame(columns=["track_id", "track_name", "artist_names", "score_cb"])
    liked_vec = M[liked_idx].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(liked_vec, M).ravel()
    df["score_cb"] = sims
    recs = df.sort_values("score_cb", ascending=False)
    recs = recs[~recs["track_id"].isin(liked_track_ids)]
    return recs[["track_id", "track_name", "artist_names", "score_cb"]].head(top_k)

def recommend_user_based(listens_df: pd.DataFrame, target_user_id: str, top_k=20):
    if "user_id" not in listens_df.columns or listens_df["user_id"].nunique() < 2:
        return pd.DataFrame(columns=["track_id", "score_cf"])
    pt = listens_df.pivot_table(index="user_id", columns="track_id",
                                values="played_at", aggfunc="count").fillna(0)
    sims = cosine_similarity(pt)
    import numpy as np
    np.fill_diagonal(sims, 0)
    sim_df = pd.DataFrame(sims, index=pt.index, columns=pt.index)
    if target_user_id not in sim_df.index:
        return pd.DataFrame(columns=["track_id", "score_cf"])
    neigh_weights = sim_df[target_user_id].sort_values(ascending=False)
    scores = (pt.T @ neigh_weights).sort_values(ascending=False)
    already = set(pt.loc[target_user_id][pt.loc[target_user_id] > 0].index)
    recs = scores.drop(labels=list(already), errors="ignore").head(top_k)
    return pd.DataFrame({"track_id": recs.index, "score_cf": recs.values})

def hybrid_recommend(items_df: pd.DataFrame, listens_df: pd.DataFrame,
                     liked_track_ids: list, target_user_id: str = None,
                     top_k: int = 20, w_cb: float = 0.6, w_cf: float = 0.4):
    cb = recommend_content_based(items_df, liked_track_ids, top_k=top_k*10)
    df = cb.copy()
    if target_user_id:
        cf = recommend_user_based(listens_df, target_user_id, top_k=top_k*10)
        if not cf.empty:
            df = df.merge(cf, on="track_id", how="left").fillna({"score_cf": 0})
            df["score"] = w_cb * df["score_cb"] + w_cf * df["score_cf"]
        else:
            df["score"] = df["score_cb"]
    else:
        df["score"] = df["score_cb"]
    return df.sort_values("score", ascending=False).head(top_k)
