import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType
from pyspark.sql.functions import col, round, max, avg, udf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configurações de conexão com o MongoDB
username = "root"
password = "mongo"
host = "localhost"
port = 27017
database_name = "spotify"
collection_name = "musicas"
auth_source = "admin"

# Verificar a conexão com o MongoDB
st.title("Análise de Dados do Spotify")
st.write("Verificando a conexão com o MongoDB...")

try:
    client = MongoClient(host=host, port=port, username=username, password=password, authSource=auth_source)
    client[auth_source].command('ping')
    st.success("Conexão ao MongoDB estabelecida com sucesso!")
except ConnectionFailure as e:
    st.error(f"Falha na conexão com o MongoDB: {e}")
    st.stop()

# Definir esquema do DataFrame
schema = StructType([
    StructField("artist_name", StringType(), True),
    StructField("track_name", StringType(), True),
    StructField("track_id", StringType(), True),
    StructField("popularity", IntegerType(), True),
    StructField("year", IntegerType(), True),
    StructField("genre", StringType(), True),
    StructField("danceability", DoubleType(), True),
    StructField("energy", DoubleType(), True),
    StructField("key", IntegerType(), True),
    StructField("loudness", DoubleType(), True),
    StructField("mode", IntegerType(), True),
    StructField("speechiness", DoubleType(), True),
    StructField("acousticness", DoubleType(), True),
    StructField("instrumentalness", DoubleType(), True),
    StructField("liveness", DoubleType(), True),
    StructField("valence", DoubleType(), True),
    StructField("tempo", DoubleType(), True),
    StructField("duration_ms", IntegerType(), True),
    StructField("time_signature", IntegerType(), True)
])

# Iniciar uma sessão Spark com o conector MongoDB
spark = SparkSession.builder \
    .appName("SpotifyAnalysis") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", f"mongodb://{username}:{password}@{host}:{port}/{database_name}.{collection_name}?authSource={auth_source}") \
    .config("spark.mongodb.output.uri", f"mongodb://{username}:{password}@{host}:{port}/{database_name}.{collection_name}?authSource={auth_source}") \
    .getOrCreate()

# Carregar os dados do MongoDB no DataFrame do PySpark
df = spark.read.format("mongo").schema(schema).load()

# Converter colunas DoubleType para FloatType
df = df.withColumn("danceability", col("danceability").cast(FloatType())) \
       .withColumn("energy", col("energy").cast(FloatType())) \
       .withColumn("loudness", col("loudness").cast(FloatType())) \
       .withColumn("speechiness", col("speechiness").cast(FloatType())) \
       .withColumn("acousticness", col("acousticness").cast(FloatType())) \
       .withColumn("instrumentalness", col("instrumentalness").cast(FloatType())) \
       .withColumn("liveness", col("liveness").cast(FloatType())) \
       .withColumn("valence", col("valence").cast(FloatType())) \
       .withColumn("tempo", col("tempo").cast(FloatType()))

# Verificar e filtrar valores inválidos em colunas numéricas
numeric_columns = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
for column in numeric_columns:
    df = df.filter(col(column).cast("float").isNotNull())

# Mostrar esquema do DataFrame e os primeiros registros para verificação
st.write("Esquema do DataFrame:")
df.printSchema()
st.write("Primeiros registros do DataFrame:")
df.show(5, truncate=False)

# Verificar se todas as colunas necessárias estão presentes
expected_columns = [
    "artist_name", "track_name", "track_id", "popularity", "year", "genre", "danceability",
    "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature"
]

missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    st.error(f"As seguintes colunas estão faltando no DataFrame: {missing_columns}")
    st.stop()

# Use uma função UDF para converter string para double
def safe_double_conversion(s):
    try:
        return float(s)
    except ValueError:
        return None  # ou escolha um valor padrão para erros de conversão

convert_to_double = udf(safe_double_conversion, DoubleType())

# Aplicando a conversão no DataFrame para colunas que precisam ser convertidas
df = df.withColumn("genre_double", convert_to_double(col("genre")))

# Filtrar os valores válidos antes do cálculo dos percentis
df = df.filter(col("valence").isNotNull() & col("energy").isNotNull())

# Certifique-se de que os valores são do tipo float
df = df.withColumn("valence", col("valence").cast(FloatType()))
df = df.withColumn("energy", col("energy").cast(FloatType()))

# Calcular percentis 99 para valence e energy
valence_99 = df.approxQuantile("valence", [0.99], 0.01)[0]
energy_99 = df.approxQuantile("energy", [0.99], 0.01)[0]

# Filtrar os dados para remover outliers extremos
df_filtered = df.filter((col("valence") <= valence_99) & (col("energy") <= energy_99))
st.write("Dados filtrados:")
df_filtered.show(5, truncate=False)

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
most_popular_per_year.show()

# Média de danceability e energy por gênero
avg_dance_energy_genre = df.groupBy('genre').agg(
    avg('danceability').alias('avg_danceability'),
    avg('energy').alias('avg_energy')
).orderBy('avg_danceability', ascending=False)
st.write("Média de danceability e energy por gênero:")
avg_dance_energy_genre.show()

# Número de músicas por ano
songs_per_year = df.groupBy('year').count().orderBy('year')
st.write("Número de músicas por ano:")
songs_per_year.show()

# Artistas mais populares
most_popular_artists = df.groupBy('artist_name').agg(avg('popularity').alias('avg_popularity')).orderBy('avg_popularity', ascending=False)
st.write("Artistas mais populares:")
most_popular_artists.show(10)

# Distribuição de tempo por gênero
avg_tempo_genre = df.groupBy('genre').agg(avg('tempo').alias('avg_tempo')).orderBy('avg_tempo', ascending=False)
st.write("Distribuição de tempo por gênero:")
avg_tempo_genre.show()

# Histograma de Popularidade
df_pandas = df.toPandas()
df_pandas['popularity'].hist(bins=30, figsize=(10, 6))
plt.title('Distribuição da Popularidade das Músicas')
plt.xlabel('Popularidade')
plt.ylabel('Frequência')
st.pyplot(plt)

# Boxplot de danceability por gênero
df_pandas.boxplot(column='danceability', by='genre', figsize=(12, 8), rot=90)
plt.title('Distribuição de Danceability por Gênero')
plt.suptitle('')
plt.xlabel('Gênero')
plt.ylabel('Danceability')
st.pyplot(plt)

# Gráfico de Dispersão de tempo vs energy
df_pandas.plot.scatter(x='tempo', y='energy', alpha=0.5, figsize=(10, 6))
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
st.pyplot(plt)

# Consulta combinada: Média de Popularidade, Danceability e Energy por Gênero
genre_stats = df.groupBy('genre').agg(
    avg('popularity').alias('avg_popularity'),
    avg('danceability').alias('avg_danceability'),
    avg('energy').alias('avg_energy')
).orderBy('avg_popularity', ascending=False)
st.write("Estatísticas por Gênero:")
genre_stats.show()

# Gráfico de barras: Média de Popularidade, Danceability e Energy por Gênero
genre_stats_pandas = genre_stats.toPandas()
genre_stats_pandas.plot.bar(x='genre', y=['avg_popularity', 'avg_danceability', 'avg_energy'], figsize=(14, 8))
plt.title('Média de Popularidade, Danceability e Energy por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Média')
st.pyplot(plt)

# Arredondar colunas tempo e energy e criar gráfico hexbin
df_rounded = df.withColumn('tempo_rounded', round(df['tempo'], -1)) \
               .withColumn('energy_rounded', round(df['energy'], 1))
st.write("Dados com colunas arredondadas:")
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
st.pyplot(plt)
