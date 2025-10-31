from auth import get_spotify_client
import pandas as pd
import time

OUTPUT_FILE = "spotify_tracks.csv"

def get_playlists(sp, user_id):
    playlists = []
    results = sp.user_playlists(user_id)
    while results:
        for item in results['items']:
            playlists.append(item)
        if results['next']:
            results = sp.next(results)
        else:
            results = None
    return playlists

def get_tracks_from_playlist(sp, playlist_id):
    tracks = []
    results = sp.playlist_items(playlist_id)
    while results:
        for item in results['items']:
            track = item['track']
            if track:
                tracks.append(track)
        if results['next']:
            results = sp.next(results)
        else:
            results = None
    return tracks

def get_audio_features(sp, track_ids):
    all_features = []
    batch_size = 50
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i+batch_size]
        try:
            features = sp.audio_features(batch)
            all_features.extend(features)
        except Exception as e:
            print(f"⚠️ Erro ao buscar features de batch {batch[:3]}...: {e}")
            all_features.extend([None]*len(batch))
        time.sleep(0.1)
    return all_features

def collect_data():
    sp = get_spotify_client()
    user_id = sp.current_user()["id"]
    all_tracks = []

    playlists = get_playlists(sp, user_id)
    print(f"📂 {len(playlists)} playlists encontradas para {user_id}")

    for playlist in playlists:
        print(f"🎵 Coletando músicas da playlist: {playlist['name']}")
        tracks = get_tracks_from_playlist(sp, playlist['id'])
        track_ids = [t['id'] for t in tracks if t['id'] is not None]
        features = get_audio_features(sp, track_ids)

        for t, f in zip(tracks, features):
            if f is None:
                continue
            all_tracks.append({
                "id": t["id"],
                "name": t["name"],
                "artist": t["artists"][0]["name"] if t["artists"] else None,
                "playlist": playlist["name"],
                **f
            })

    df = pd.DataFrame(all_tracks)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Dados salvos em {OUTPUT_FILE}")

if __name__ == "__main__":
    collect_data()
