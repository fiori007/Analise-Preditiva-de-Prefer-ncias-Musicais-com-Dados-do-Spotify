import pandas as pd

def load_raw(raw_dir: str = "data/raw"):
    tracks = pd.read_csv(f"{raw_dir}/tracks.csv")
    artists = pd.read_csv(f"{raw_dir}/artists.csv")
    listens = pd.read_csv(f"{raw_dir}/listens.csv", parse_dates=["played_at"])
    return tracks, artists, listens

def preprocess(tracks, artists, listens):
    tracks = tracks.drop_duplicates("track_id")
    artists = artists.drop_duplicates("artist_id")
    listens = listens.drop_duplicates()

    listens["played_at"] = pd.to_datetime(listens["played_at"], errors="coerce")

    df = tracks.copy()
    df["primary_artist_id"] = tracks["artist_ids"].str.split(",").str[0]
    df = df.merge(artists.add_prefix("artist_"),
                  left_on="primary_artist_id",
                  right_on="artist_artist_id",
                  how="left")
    return df, listens
