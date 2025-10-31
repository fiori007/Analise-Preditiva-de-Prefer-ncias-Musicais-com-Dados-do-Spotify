import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de visualização
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_data():
    df = pd.read_csv("spotify_tracks.csv")
    print("✅ Dados carregados com sucesso!")
    print(df.head())
    print("\n📊 Informações gerais:")
    print(df.info())
    return df

def summary_statistics(df):
    print("\n📈 Estatísticas descritivas:")
    print(df.describe(include="all"))

def plot_popularity_distribution(df):
    plt.figure()
    sns.histplot(df["popularity"], bins=20, kde=True, color="blue")
    plt.title("Distribuição da Popularidade das Músicas")
    plt.xlabel("Popularidade")
    plt.ylabel("Frequência")
    plt.savefig("plots/popularity_distribution.png")
    plt.show()

def plot_audio_features(df):
    features = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness", "tempo"]

    for feature in features:
        if feature in df.columns:
            plt.figure()
            sns.histplot(df[feature], bins=20, kde=True, color="green")
            plt.title(f"Distribuição de {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequência")
            plt.savefig(f"plots/{feature}_distribution.png")
            plt.show()

def correlation_heatmap(df):
    plt.figure()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de Correlação das Features de Áudio")
    plt.savefig("plots/correlation_heatmap.png")
    plt.show()

if __name__ == "__main__":
    # Criar pasta para salvar gráficos
    import os
    if not os.path.exists("plots"):
        os.makedirs("plots")

    df = load_data()
    summary_statistics(df)
    plot_popularity_distribution(df)
    plot_audio_features(df)
    correlation_heatmap(df)
