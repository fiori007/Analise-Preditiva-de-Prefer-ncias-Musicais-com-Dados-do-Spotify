from typing import List, Dict
import time
import spotipy
import pandas as pd
from spotipy import Spotify


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def get_current_user(sp: Spotify) -> Dict:
    u = sp.current_user()
    return {
        "user_id": u["id"],
        "display_name": u.get("display_name"),
        "country": u.get("country"),
        "product": u.get("product"),
    }


def get_user_top(sp: Spotify,
                 entity: str = "tracks",
                 time_range: str = "medium_term",
                 limit: int = 50,
                 offset: int = 100) -> List[Dict]:
    if entity == "tracks":
        return sp.current_user_top_tracks(
            limit=limit,
            offset=offset,
            time_range=time_range
        ).get("items", [])
    else:
        return sp.current_user_top_artists(
            limit=limit,
            offset=offset,
            time_range=time_range
        ).get("items", [])


def get_recently_played(sp: Spotify, limit: int = 50):
    return sp.current_user_recently_played(limit=limit).get("items", [])


def get_tracks(sp: Spotify, track_ids: List[str]) -> List[Dict]:
    out = []
    for batch in chunked(track_ids, 50):
        res = sp.tracks(batch)
        out.extend(res.get("tracks", []))
        time.sleep(0.1)
    return out


def get_artists(sp: Spotify, artist_ids: List[str]) -> List[Dict]:
    out = []
    for batch in chunked(artist_ids, 50):
        res = sp.artists(batch)
        out.extend(res.get("artists", []))
        time.sleep(0.1)
    return out


def normalize_tracks(raw_tracks: List[Dict]) -> pd.DataFrame:
    rows = []
    for t in raw_tracks:
        if not t:
            continue
        rows.append({
            "track_id": t.get("id"),
            "track_name": t.get("name"),
            "album_id": t.get("album", {}).get("id"),
            "album_name": t.get("album", {}).get("name"),
            "album_release_date": t.get("album", {}).get("release_date"),
            "artist_ids": ",".join([a.get("id") for a in t.get("artists", []) if a.get("id")]),
            "artist_names": ",".join([a.get("name") for a in t.get("artists", []) if a.get("name")]),
            "duration_ms": t.get("duration_ms"),
            "popularity": t.get("popularity"),
            "explicit": t.get("explicit"),
        })
    return pd.DataFrame(rows).drop_duplicates("track_id")


def normalize_artists(raw_artists: List[Dict]) -> pd.DataFrame:
    rows = []
    for a in raw_artists:
        if not a:
            continue
        rows.append({
            "artist_id": a.get("id"),
            "artist_name": a.get("name"),
            "genres": ",".join(a.get("genres", [])),
            "followers": a.get("followers", {}).get("total"),
            "popularity_artist": a.get("popularity"),
        })
    return pd.DataFrame(rows).drop_duplicates("artist_id")


def normalize_listens(recent_items: List[Dict], user_id: str) -> pd.DataFrame:
    rows = []
    for it in recent_items:
        t = it.get("track", {})
        rows.append({
            "user_id": user_id,
            "played_at": it.get("played_at"),
            "track_id": t.get("id"),
            "artist_ids": ",".join([a.get("id") for a in t.get("artists", []) if a.get("id")]),
            "context_type": (it.get("context") or {}).get("type"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce")
    return df
