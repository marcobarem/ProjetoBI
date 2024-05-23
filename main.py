# main.py

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, max, round
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Iniciar uma sessão Spark com o conector MongoDB
spark = SparkSession.builder \
    .appName("SpotifyAnalysis") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", f"mongodb://admin:pass@localhost:27017/{database_name}.{collection_name}?authSource={auth_source}") \
    .config("spark.mongodb.output.uri", f"mongodb://admin:pass@localhost:27017/{database_name}.{collection_name}?authSource={auth_source}") \
    .getOrCreate()

# Carregar os dados do MongoDB no DataFrame do PySpark
df = spark.read.format("mongo").load()

# Mostrar os primeiros registros do DataFrame
print("Primeiros registros do DataFrame:")
df.show(5)

# Estatísticas descritivas
print("Estatísticas descritivas:")
df.describe().show()

# Calcular percentis 99 para valence e energy
valence_99 = df.approxQuantile("valence", [0.99], 0.01)[0]
energy_99 = df.approxQuantile("energy", [0.99], 0.01)[0]

# Filtrar os dados para remover outliers extremos
df_filtered = df.filter((col("valence") <= valence_99) & (col("energy") <= energy_99))
print("Dados filtrados:")
df_filtered.show(5)

# Converter para Pandas DataFrame para visualização
df_filtered_pandas = df_filtered.toPandas()

# Criar gráfico hexbin
plt.figure(figsize=(10, 6))
hb = plt.hexbin(df_filtered_pandas['valence'], df_filtered_pandas['energy'], gridsize=50, cmap='Blues', mincnt=1)
plt.colorbar(hb, label='Contagem no Bin')
plt.title('Análise de Sentimento Musical - Hexbin Plot com Dados Filtrados')
plt.xlabel('Valence (Positividade)')
plt.ylabel('Energy (Intensidade)')
plt.grid(True)
plt.savefig('hexbin_plot.png')
plt.show()

# Músicas mais populares por ano
most_popular_per_year = df.groupBy('year').agg(max('popularity').alias('max_popularity')).orderBy('year')
print("Músicas mais populares por ano:")
most_popular_per_year.show()

# Média de danceability e energy por gênero
avg_dance_energy_genre = df.groupBy('genre').agg(
    avg('danceability').alias('avg_danceability'),
    avg('energy').alias('avg_energy')
).orderBy('avg_danceability', ascending=False)
print("Média de danceability e energy por gênero:")
avg_dance_energy_genre.show()

# Número de músicas por ano
songs_per_year = df.groupBy('year').count().orderBy('year')
print("Número de músicas por ano:")
songs_per_year.show()

# Artistas mais populares
most_popular_artists = df.groupBy('artist_name').agg(avg('popularity').alias('avg_popularity')).orderBy('avg_popularity', ascending=False)
print("Artistas mais populares:")
most_popular_artists.show(10)

# Distribuição de tempo por gênero
avg_tempo_genre = df.groupBy('genre').agg(avg('tempo').alias('avg_tempo')).orderBy('avg_tempo', ascending=False)
print("Distribuição de tempo por gênero:")
avg_tempo_genre.show()

# Histograma de Popularidade
df_pandas = df.toPandas()
df_pandas['popularity'].hist(bins=30, figsize=(10, 6))
plt.title('Distribuição da Popularidade das Músicas')
plt.xlabel('Popularidade')
plt.ylabel('Frequência')
plt.savefig('popularity_histogram.png')
plt.show()

# Boxplot de danceability por gênero
df_pandas.boxplot(column='danceability', by='genre', figsize=(12, 8), rot=90)
plt.title('Distribuição de Danceability por Gênero')
plt.suptitle('')
plt.xlabel('Gênero')
plt.ylabel('Danceability')
plt.savefig('danceability_boxplot.png')
plt.show()

# Gráfico de Dispersão de tempo vs energy
df_pandas.plot.scatter(x='tempo', y='energy', alpha=0.5, figsize=(10, 6))
plt.title('Relação entre Tempo e Energy')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Energy')
plt.grid(True)
plt.savefig('tempo_vs_energy_scatter.png')
plt.show()

# Heatmap de Correlações entre Atributos
corr = df_pandas.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor das Correlações')
plt.savefig('correlation_heatmap.png')
plt.show()

# Linha do Tempo das Músicas mais Populares por Ano
most_popular_per_year_pandas = most_popular_per_year.toPandas()
plt.figure(figsize=(12, 6))
plt.plot(most_popular_per_year_pandas['year'], most_popular_per_year_pandas['max_popularity'], marker='o')
plt.title('Popularidade Máxima das Músicas por Ano')
plt.xlabel('Ano')
plt.ylabel('Popularidade Máxima')
plt.grid(True)
plt.savefig('popularity_over_time.png')
plt.show()

# Consulta combinada: Média de Popularidade, Danceability e Energy por Gênero
genre_stats = df.groupBy('genre').agg(
    avg('popularity').alias('avg_popularity'),
    avg('danceability').alias('avg_danceability'),
    avg('energy').alias('avg_energy')
).orderBy('avg_popularity', ascending=False)
genre_stats.show()

# Gráfico de barras: Média de Popularidade, Danceability e Energy por Gênero
genre_stats_pandas = genre_stats.toPandas()
genre_stats_pandas.plot.bar(x='genre', y=['avg_popularity', 'avg_danceability', 'avg_energy'], figsize=(14, 8))
plt.title('Média de Popularidade, Danceability e Energy por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Média')
plt.savefig('genre_stats.png')
plt.show()

# Arredondar colunas tempo e energy e criar gráfico hexbin
df_rounded = df.withColumn('tempo_rounded', round(df['tempo'], -1)) \
               .withColumn('energy_rounded', round(df['energy'], 1))
df_rounded.show(5)

# Converter para Pandas DataFrame e criar gráfico hexbin
df_rounded_pandas = df_rounded.select('tempo_rounded', 'energy_rounded').toPandas()

plt.figure(figsize=(10, 6))
hb = plt.hexbin(df_rounded_pandas['tempo_rounded'], df_rounded_pandas['energy_rounded'], gridsize=30, cmap='Greens', mincnt=1)
plt.colorbar(hb, label='Contagem no Bin')
plt.title('Distribuição de Tempo e Energy')
plt.xlabel('Tempo (BPM arredondado)')
plt.ylabel('Energy (arredondado)')
plt.grid(True)
plt.savefig('tempo_energy_hexbin.png')
plt.show()
