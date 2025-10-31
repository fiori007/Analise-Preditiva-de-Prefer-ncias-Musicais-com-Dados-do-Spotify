import argparse
import pandas as pd
from pathlib import Path
from src.auth import get_spotify_client
from src.collect import (
    get_current_user, get_user_top, get_recently_played,
    get_tracks, get_artists,
    normalize_tracks, normalize_artists, normalize_listens
)
from src.preprocess import load_raw, preprocess
from src.features import build_preference_labels, add_temporal_features, user_profile_features
from src.model import train_and_eval
from src.recommend import hybrid_recommend


RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
PRO = Path("data/processed")
PRO.mkdir(parents=True, exist_ok=True)


def cmd_auth():
    sp = get_spotify_client()
    me = sp.current_user()
    print("Autenticado como:", me.get("display_name"), me.get("id"))


def cmd_collect():
    sp = get_spotify_client()
    me = get_current_user(sp)
    user_id = me["user_id"]
    print("Coletando para:", user_id)

    # Passando offset=100 aqui para top endpoints
    top_tracks = get_user_top(sp, entity="tracks", time_range="medium_term", limit=50, offset=100)
    top_artists = get_user_top(sp, entity="artists", time_range="medium_term", limit=50, offset=100)
    recents = get_recently_played(sp, limit=50)

    track_ids = list({t["id"] for t in top_tracks if t})
    artist_ids = list({a["id"] for a in top_artists if a})

    tracks_full = get_tracks(sp, track_ids)
    artists_full = get_artists(sp, artist_ids)

    df_tracks = normalize_tracks(tracks_full)
    df_art = normalize_artists(artists_full)
    df_listen = normalize_listens(recents, user_id=user_id)

    # Salva apenas os dados disponíveis (sem audio_features)
    df_tracks.to_csv(RAW / "tracks.csv", index=False)
    df_art.to_csv(RAW / "artists.csv", index=False)
    if not df_listen.empty:
        df_listen.to_csv(RAW / "listens.csv", index=False)

    print("Arquivos salvos em data/raw/ (sem audio_features)")


def cmd_preprocess():
    tracks, artists, listens = load_raw()
    items, listens_clean = preprocess(tracks, artists, listens)
    items.to_csv(PRO / "items.csv", index=False)
    listens_clean.to_csv(PRO / "listens.csv", index=False)
    print("Pré-processamento concluído e salvo em data/processed/")


def cmd_features(playcount_threshold: int):
    items = pd.read_csv(PRO / "items.csv")
    listens = pd.read_csv(PRO / "listens.csv", parse_dates=["played_at"])

    items_labeled = build_preference_labels(items, listens, playcount_threshold=playcount_threshold)
    listens_t = add_temporal_features(listens)

    profile = user_profile_features(items_labeled)

    items_labeled.to_csv(PRO / "items_labeled.csv", index=False)
    listens_t.to_csv(PRO / "listens_temporal.csv", index=False)
    pd.Series(profile).to_csv(PRO / "user_profile.csv")

    print("Features geradas e salvas em data/processed/")


def cmd_train():
    train_and_eval(str(PRO / "items_labeled.csv"))
    print("Treinamento e avaliação concluídos.")


def cmd_recommend(liked_track_ids: str, target_user_id: str, top_k: int):
    items = pd.read_csv(PRO / "items_labeled.csv")
    listens = pd.read_csv(PRO / "listens_temporal.csv", parse_dates=["played_at"])

    liked_list = [tid.strip() for tid in liked_track_ids.split(",") if tid.strip()]
    recs = hybrid_recommend(items, listens, liked_list, target_user_id, top_k=top_k)
    print(recs.head(top_k))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth")
    sub.add_parser("collect")
    sub.add_parser("preprocess")

    p_feat = sub.add_parser("features")
    p_feat.add_argument("--playcount-threshold", type=int, default=2)

    sub.add_parser("train")

    p_rec = sub.add_parser("recommend")
    p_rec.add_argument("--liked-track-ids", type=str, default="")
    p_rec.add_argument("--target-user-id", type=str, default=None)
    p_rec.add_argument("--top-k", type=int, default=20)

    args = ap.parse_args()

    if args.cmd == "auth":
        cmd_auth()
    elif args.cmd == "collect":
        cmd_collect()
    elif args.cmd == "preprocess":
        cmd_preprocess()
    elif args.cmd == "features":
        cmd_features(playcount_threshold=args.playcount_threshold)
    elif args.cmd == "train":
        cmd_train()
    elif args.cmd == "recommend":
        cmd_recommend(
            liked_track_ids=args.liked_track_ids,
            target_user_id=args.target_user_id,
            top_k=args.top_k
        )
