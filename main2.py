# main2.py

import streamlit as st
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
    client[auth_source].command('ping')
    st.write("Conexão ao MongoDB estabelecida com sucesso!")
except ConnectionFailure as e:
    st.write(f"Falha na conexão com o MongoDB: {e}")

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
st.write("Primeiros registros do DataFrame:")
st.write(df.limit(5).toPandas())

# Estatísticas descritivas
st.write("Estatísticas descritivas:")
st.write(df.describe().toPandas())

# Calcular percentis 99 para valence e energy
valence_99 = df.approxQuantile("valence", [0.99], 0.01)[0]
energy_99 = df.approxQuantile("energy", [0.99], 0.01)[0]

# Filtrar os dados para remover outliers extremos
df_filtered = df.filter((col("valence") <= valence_99) & (col("energy") <= energy_99))
st.write("Dados filtrados:")
st.write(df_filtered.limit(5).toPandas())

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
st.pyplot(plt)

# Músicas mais populares por ano
most_popular_per_year = df.groupBy('year').agg(max('popularity').alias('max_popularity')).orderBy('year')
st.write("Músicas mais populares por ano:")
st.write(most_popular_per_year.toPandas())

# Média de danceability e energy por gênero
avg_dance_energy_genre = df.groupBy('genre').agg(
    avg('danceability').alias('avg_danceability'),
    avg('energy').alias('avg_energy')
).orderBy('avg_danceability', ascending=False)
st.write("Média de danceability e energy por gênero:")
st.write(avg_dance_energy_genre.toPandas())

# Número de músicas por ano
songs_per_year = df.groupBy('year').count().orderBy('year')
st.write("Número de músicas por ano:")
st.write(songs_per_year.toPandas())

# Artistas mais populares
most_popular_artists = df.groupBy('artist_name').agg(avg('popularity').alias('avg_popularity')).orderBy('avg_popularity', ascending=False)
st.write("Artistas mais populares:")
st.write(most_popular_artists.limit(10).toPandas())

# Distribuição de tempo por gênero
avg_tempo_genre = df.groupBy('genre').agg(avg('tempo').alias('avg_tempo')).orderBy('avg_tempo', ascending=False)
st.write("Distribuição de tempo por gênero:")
st.write(avg_tempo_genre.toPandas())

# Histograma de Popularidade
df_pandas = df.toPandas()
plt.figure(figsize=(10, 6))
df_pandas['popularity'].hist(bins=30)
plt.title('Distribuição da Popularidade das Músicas')
plt.xlabel('Popularidade')
plt.ylabel('Frequência')
st.pyplot(plt)

# Boxplot de danceability por gênero
plt.figure(figsize=(12, 8))
df_pandas.boxplot(column='danceability', by='genre', rot=90)
plt.title('Distribuição de Danceability por Gênero')
plt.suptitle('')
plt.xlabel('Gênero')
plt.ylabel('Danceability')
st.pyplot(plt)

# Gráfico de Dispersão de tempo vs energy
plt.figure(figsize=(10, 6))
plt.scatter(df_pandas['tempo'], df_pandas['energy'], alpha=0.5)
plt.title('Relação entre Tempo e Energy')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Energy')
plt.grid(True)
st.pyplot(plt)

# Heatmap de Correlações entre Atributos
corr = df_pandas.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor das Correlações')
st.pyplot(plt)

# Linha do Tempo das Músicas mais Populares por Ano
most_popular_per_year_pandas = most_popular_per_year.toPandas()
plt.figure(figsize=(12, 6))
plt.plot(most_popular_per_year_pandas['year'], most_popular_per_year_pandas['max_popularity'], marker='o')
plt.title('Popularidade Máxima das Músicas por Ano')
plt.xlabel('Ano')
plt.ylabel('Popularidade Máxima')
plt.grid(True)
