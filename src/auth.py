import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

scope = (
    "user-library-read "
    "playlist-read-private "
    "user-top-read "
    "user-read-recently-played"
)

def get_user_client():
    """Autenticação com o usuário logado."""
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=scope,
        cache_path=".cache"
    ))

def get_app_client():
    """Autenticação com o app (sem usuário)."""
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
    ))

def get_spotify_client():
    """Cliente padrão com OAuth completo."""
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=os.getenv("SPOTIFY_SCOPES"),
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        cache_path=".cache"
    ))
