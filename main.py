from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import redis

# Configurações de conexão com o MongoDB
mongo_uri = "mongodb://admin:pass@localhost:27017/"
database_name = "spotify"
collection_name = "musicas"
auth_source = "admin"

# Verificar a conexão com o MongoDB
try:
    client = MongoClient(mongo_uri, authSource=auth_source)
    client.admin.command('ping')
    print("Conexão ao MongoDB estabelecida com sucesso!")
except ConnectionFailure as e:
    print(f"Falha na conexão com o MongoDB: {e}")
except OperationFailure as e:
    print(f"Falha na autenticação com o MongoDB: {e}")

# Carregar os dados do MongoDB no DataFrame do pandas
db = client[database_name]
collection = db[collection_name]
data = list(collection.find())
df = pd.DataFrame(data)

# Mostrar os primeiros registros do DataFrame
print("Primeiros registros do DataFrame:")
print(df.head())

# Estatísticas descritivas
print("Estatísticas descritivas:")
print(df.describe())

# Calcular percentis 99 para valence e energy
valence_99 = df['valence'].quantile(0.99)
energy_99 = df['energy'].quantile(0.99)

# Filtrar os dados para remover outliers extremos
df_filtered = df[(df['valence'] <= valence_99) & (df['energy'] <= energy_99)]
print("Dados filtrados:")
print(df_filtered.head())

# Criar gráfico hexbin
plt.figure(figsize=(10, 6))
hb = plt.hexbin(df_filtered['valence'], df_filtered['energy'], gridsize=50, cmap='Blues', mincnt=1)
plt.colorbar(hb, label='Contagem no Bin')
plt.title('Análise de Sentimento Musical - Hexbin Plot com Dados Filtrados')
plt.xlabel('Valence (Positividade)')
plt.ylabel('Energy (Intensidade)')
plt.grid(True)
plt.savefig('hexbin_plot.png')
plt.show()

# Músicas mais populares por ano
most_popular_per_year = df.groupby('year')['popularity'].max().reset_index()
print("Músicas mais populares por ano:")
print(most_popular_per_year)

# Média de danceability e energy por gênero
avg_dance_energy_genre = df.groupby('genre')[['danceability', 'energy']].mean().reset_index()
print("Média de danceability e energy por gênero:")
print(avg_dance_energy_genre)

# Número de músicas por ano
songs_per_year = df.groupby('year').size().reset_index(name='count')
print("Número de músicas por ano:")
print(songs_per_year)

# Artistas mais populares
most_popular_artists = df.groupby('artist_name')['popularity'].mean().reset_index().sort_values(by='popularity', ascending=False)
print("Artistas mais populares:")
print(most_popular_artists.head(10))

# Distribuição de tempo por gênero
avg_tempo_genre = df.groupby('genre')['tempo'].mean().reset_index().sort_values(by='tempo', ascending=False)
print("Distribuição de tempo por gênero:")
print(avg_tempo_genre)

# Histograma de Popularidade
df['popularity'].hist(bins=30, figsize=(10, 6))
plt.title('Distribuição da Popularidade das Músicas')
plt.xlabel('Popularidade')
plt.ylabel('Frequência')
plt.savefig('popularity_histogram.png')
plt.show()

# Boxplot de danceability por gênero
df.boxplot(column='danceability', by='genre', figsize=(12, 8), rot=90)
plt.title('Distribuição de Danceability por Gênero')
plt.suptitle('')
plt.xlabel('Gênero')
plt.ylabel('Danceability')
plt.savefig('danceability_boxplot.png')
plt.show()

# Gráfico de Dispersão de tempo vs energy
df.plot.scatter(x='tempo', y='energy', alpha=0.5, figsize=(10, 6))
plt.title('Relação entre Tempo e Energy')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Energy')
plt.grid(True)
plt.savefig('tempo_vs_energy_scatter.png')
plt.show()

# Heatmap de Correlações entre Atributos
corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor das Correlações')
plt.savefig('correlation_heatmap.png')
plt.show()

# Linha do Tempo das Músicas mais Populares por Ano
plt.figure(figsize=(12, 6))
plt.plot(most_popular_per_year['year'], most_popular_per_year['popularity'], marker='o')
plt.title('Popularidade Máxima das Músicas por Ano')
plt.xlabel('Ano')
plt.ylabel('Popularidade Máxima')
plt.grid(True)
plt.savefig('popularity_over_time.png')
plt.show()

# Consulta combinada: Média de Popularidade, Danceability e Energy por Gênero
genre_stats = df.groupby('genre')[['popularity', 'danceability', 'energy']].mean().reset_index().sort_values(by='popularity', ascending=False)
print("Média de Popularidade, Danceability e Energy por Gênero:")
print(genre_stats)

# Gráfico de barras: Média de Popularidade, Danceability e Energy por Gênero
genre_stats.plot.bar(x='genre', y=['popularity', 'danceability', 'energy'], figsize=(14, 8))
plt.title('Média de Popularidade, Danceability e Energy por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Média')
plt.savefig('genre_stats.png')
plt.show()

# Arredondar colunas tempo e energy e criar gráfico hexbin
df['tempo_rounded'] = df['tempo'].round(-1)
df['energy_rounded'] = df['energy'].round(1)
print("Dados arredondados:")
print(df[['tempo_rounded', 'energy_rounded']].head())

# Criar gráfico hexbin com dados arredondados
plt.figure(figsize=(10, 6))
hb = plt.hexbin(df['tempo_rounded'], df['energy_rounded'], gridsize=30, cmap='Greens', mincnt=1)
plt.colorbar(hb, label='Contagem no Bin')
plt.title('Distribuição de Tempo e Energy')
plt.xlabel('Tempo (BPM arredondado)')
plt.ylabel('Energy (arredondado)')
plt.grid(True)
plt.savefig('tempo_energy_hexbin.png')
plt.show()
