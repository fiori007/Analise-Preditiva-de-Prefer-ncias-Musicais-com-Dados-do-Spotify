from src.auth import get_spotify_client

def main():
    sp = get_spotify_client()
    # aqui usamos um track ID público conhecido
    track_id = "4uLU6hMCjMI75M1A2tKUQC"  # exemplo: “Bohemian Rhapsody” do Queen
    try:
        features = sp.audio_features([track_id])
        print("features:", features)
    except Exception as e:
        print("Erro ao chamar audio_features:", e)

if __name__ == "__main__":
    main()
