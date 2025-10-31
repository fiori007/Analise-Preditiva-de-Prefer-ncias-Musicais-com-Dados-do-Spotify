import pandas as pd
from pathlib import Path

# Paths
RAW_DIR  = Path("data/raw")
PRO_DIR  = Path("data/processed")

# Arquivos de entrada
tracks_path   = RAW_DIR  / "tracks.csv"
artists_path  = RAW_DIR  / "artists.csv"
listens_path  = RAW_DIR  / "listens.csv"

# Arquivos de saída
items_labeled_path = PRO_DIR / "items_labeled.csv"

def main():
    # Carrega os dados que temos
    df_tracks  = pd.read_csv(tracks_path)
    df_artists = pd.read_csv(artists_path)
    df_listens = pd.read_csv(listens_path, parse_dates=["played_at"], infer_datetime_format=True)

    # Merge básico: tracks + artistas
    df_tracks = df_tracks.drop_duplicates("track_id")
    df_artists = df_artists.drop_duplicates("artist_id")

    df_tracks["primary_artist_id"] = df_tracks["artist_ids"].str.split(",").str[0]
    df_items = df_tracks.merge(
        df_artists.add_prefix("artist_"),
        left_on="primary_artist_id",
        right_on="artist_artist_id",
        how="left"
    )

    # Adiciona coluna playcount
    counts = df_listens.groupby("track_id").size().rename("playcount").reset_index()
    df_items = df_items.merge(counts, on="track_id", how="left").fillna({"playcount": 0})
    df_items["preferred"] = (df_items["playcount"] >= 2).astype(int)

    # Salva
    PRO_DIR.mkdir(parents=True, exist_ok=True)
    df_items.to_csv(items_labeled_path, index=False)
    print(f"✅ Arquivo gerado: {items_labeled_path} (sem audio_features)")

if __name__ == "__main__":
    main()
