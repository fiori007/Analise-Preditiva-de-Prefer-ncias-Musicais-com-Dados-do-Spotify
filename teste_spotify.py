from src.auth import make_spotify_client

# Cria o cliente Spotify autenticado
sp = make_spotify_client()

# Pega informações do usuário logado
user = sp.current_user()
print("Usuário autenticado:", user['display_name'])

# Lista as 5 primeiras playlists do usuário
playlists = sp.current_user_playlists(limit=5)
print("\nSuas playlists:")
for i, playlist in enumerate(playlists['items'], start=1):
    print(f"{i}. {playlist['name']} ({playlist['tracks']['total']} faixas)")
