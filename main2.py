import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.sql.functions import col

# Configurar Streamlit
st.set_page_config(layout="wide")

# Título e Informações Gerais
st.title("Análise de Dados Musicais do Spotify")
st.markdown("""
### Integrantes: [Marco Barem]
### Data: [28/05]
### Disciplinas: [Inteligencia de Negócios e Big Data]
""")
st.markdown("## Descrição do Estudo")
st.write("Este estudo analisa a popularidade das músicas no Spotify ao longo dos anos, examinando diversos fatores como gênero, artista, características musicais, e outros.")

# Conectar ao MongoDB
try:
    client = MongoClient("mongodb://root:mongo@localhost:27017", serverSelectionTimeoutMS=5000)
    client.server_info()  # Isso lançará uma exceção se não puder se conectar ao servidor.
    st.success("Conexão estabelecida com sucesso!")
except ConnectionFailure:
    st.error("Falha na conexão ao servidor MongoDB")

# Selecionar o banco de dados
db = client['spotify']

# Selecionar a coleção
collection = db['musicas']

# Iniciar sessão do Spark
spark = SparkSession.builder.appName("SpotifyAnalysis").getOrCreate()

# Definir esquema
schema = StructType([
    StructField("popularity", DoubleType(), True),
    StructField("year", DoubleType(), True),
    StructField("genre", StringType(), True),
    StructField("artist_name", StringType(), True),
    StructField("danceability", DoubleType(), True),
    StructField("energy", DoubleType(), True),
    StructField("acousticness", DoubleType(), True),
    StructField("speechiness", DoubleType(), True),
    StructField("instrumentalness", DoubleType(), True),
    StructField("duration_ms", DoubleType(), True),
    StructField("time_signature", DoubleType(), True),
    StructField("loudness", DoubleType(), True)
])

# Função para carregar dados
def load_data():
    data = list(collection.find({}, {'_id': 0, 'popularity': 1, 'year': 1, 'genre': 1, 'artist_name': 1, 'danceability': 1, 'energy': 1, 'acousticness': 1, 'speechiness': 1, 'instrumentalness': 1, 'duration_ms': 1, 'time_signature': 1, 'loudness': 1}))
    for item in data:
        for key in item:
            try:
                item[key] = float(item[key])
            except (ValueError, TypeError):
                item[key] = None
    return spark.createDataFrame(data, schema=schema)

df = load_data()

# Verificar se dados foram carregados
if df.count() == 0:
    st.error("Nenhum dado foi carregado no DataFrame.")
else:
    st.success(f"{df.count()} registros carregados no DataFrame.")

# Remover valores nulos
df = df.na.drop()

# Verificar os dados carregados
st.write("Dados Carregados (primeiros 5 registros):")
st.write(df.limit(5).toPandas())  # Converter e mostrar os primeiros 5 registros usando Pandas

# Pergunta 1: Como a popularidade das Músicas mudou ao longo dos anos?
st.markdown("## Pergunta 1: Como a popularidade das Músicas mudou ao longo dos anos?")
df_popularity_year = df.filter((df.year >= 2000) & (df.year <= 2023))

# Verificar o DataFrame filtrado
st.write("Dados Filtrados por Ano (primeiros 5 registros):")
st.write(df_popularity_year.limit(5).toPandas())  # Converter e mostrar os primeiros 5 registros usando Pandas

# Agrupar por ano e gênero e calcular a média de popularidade
df_grouped = df_popularity_year.groupBy('year', 'genre').avg('popularity').orderBy('year')

# Verificar os dados agrupados
st.write("Dados Agrupados (primeiros 5 registros):")
st.write(df_grouped.limit(5).toPandas())  # Converter e mostrar os primeiros 5 registros usando Pandas

# Converter para Pandas
genre_popularity = df_grouped.toPandas()

# Verificar os dados convertidos para Pandas
st.write("Dados Convertidos para Pandas (primeiros 5 registros):")
st.write(genre_popularity.head())

# Plotar o gráfico
fig1, ax1 = plt.subplots(figsize=(14, 10))
sns.lineplot(data=genre_popularity, x='year', y='avg(popularity)', hue='genre', ax=ax1)
ax1.set_title('Média de Popularidade das Músicas por Gênero ao Longo dos Anos (2000 em diante)')
ax1.set_xlabel('Ano')
ax1.set_ylabel('Média de Popularidade')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)  # Ajuste da legenda para duas colunas
st.pyplot(fig1)

st.markdown("""
## Conclusão
### Como a popularidade das Músicas mudou ao longo dos anos?

O gráfico mostra que a popularidade das músicas, em média, tem aumentado ao longo dos anos. Gêneros como Pop, Rock e Dance têm mantido uma popularidade alta constante, enquanto outros gêneros apresentam variações. Isso indica uma tendência de crescimento na aceitação de uma variedade de gêneros musicais ao longo do tempo.
""")

# Pergunta 2: Quais gêneros são mais populares?
st.markdown("## Pergunta 2: Quais gêneros são mais populares?")
top_genres = df.groupBy('genre').avg('popularity').orderBy('avg(popularity)', ascending=False).limit(20)
top_genres_pd = top_genres.toPandas()
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.barplot(data=top_genres_pd, x='avg(popularity)', y='genre', ax=ax2)
ax2.set_title('Top 20 Gêneros Musicais por Popularidade Média')
ax2.set_xlabel('Popularidade')
ax2.set_ylabel('Gênero')
st.pyplot(fig2)

st.markdown("""
## Conclusão
### Quais gêneros são mais populares?

Os gêneros mais populares são Pop, Rock e Dance, com o Pop destacando-se como o gênero de maior popularidade média. Outros gêneros como Metal, Sad, e Folk também mostram alta popularidade, indicando uma diversidade de preferências musicais entre os ouvintes.
""")

# Pergunta 3: Quais artistas têm a maior quantidade de músicas populares?
st.markdown("## Pergunta 3: Quais artistas têm a maior quantidade de músicas populares?")
top_artists = df.groupBy('artist_name').count().orderBy('count', ascending=False).limit(20)
top_artists_pd = top_artists.toPandas()
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.barplot(data=top_artists_pd, x='count', y='artist_name', ax=ax3)
ax3.set_title('Top 20 Artistas com Mais Músicas Populares')
ax3.set_xlabel('Número de Músicas Populares')
ax3.set_ylabel('Artista')
st.pyplot(fig3)

st.markdown("""
## Conclusão
### Quais artistas têm a maior quantidade de músicas populares?

Artistas como Hans Zimmer, Glee Cast, e Pritam lideram em número de músicas populares. A presença de uma mistura de artistas de trilhas sonoras, bandas e artistas solo mostra a diversidade nas preferências dos ouvintes.
""")

# Pergunta 4: Existe uma correlação entre danceabilidade e a popularidade das músicas?
st.markdown("## Pergunta 4: Existe uma correlação entre danceabilidade e a popularidade das músicas?")
correlation_danceability = df.corr('danceability', 'popularity')
st.write(f'Correlação entre Danceabilidade e Popularidade: {correlation_danceability}')

sample_df = df.sample(False, 0.1).toPandas()  # Amostra de 10% dos dados
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='danceability', y='popularity', data=sample_df, alpha=0.7, ax=ax4)
sns.regplot(x='danceability', y='popularity', data=sample_df, scatter=False, color='red', ax=ax4)
ax4.set_xlabel('Danceabilidade')
ax4.set_ylabel('Popularidade')
ax4.set_title('Correlação entre Danceabilidade e Popularidade')
st.pyplot(fig4)

st.markdown("""
## Conclusão
### Existe uma correlação entre danceabilidade e a popularidade das músicas?

A correlação entre danceabilidade e popularidade é fraca, sugerindo que a capacidade de uma música para dançar não é um fator significativo na determinação de sua popularidade.
""")



# Pergunta 5: Como a energia das músicas influencia sua popularidade?
st.markdown("## Pergunta 5: Como a energia das músicas influencia sua popularidade?")
correlation_energy = df.corr('energy', 'popularity')
st.write(f'Correlação entre Energia e Popularidade: {correlation_energy}')

fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='energy', y='popularity', data=sample_df, alpha=0.7, ax=ax5)
sns.regplot(x='energy', y='popularity', data=sample_df, scatter=False, color='red', ax=ax5)
ax5.set_xlabel('Energia')
ax5.set_ylabel('Popularidade')
ax5.set_title('Correlação entre Energia e Popularidade')
st.pyplot(fig5)

st.markdown("""
## Conclusão
### Como a energia das músicas influencia sua popularidade?

A correlação entre energia e popularidade é praticamente inexistente, indicando que a energia de uma música não tem um impacto significativo na sua popularidade.
""")

# Pergunta 6: Quais são as características comuns das músicas mais populares em termos de acousticness, speechiness e instrumentalness?
st.markdown("## Pergunta 6: Quais são as características comuns das músicas mais populares em termos de acousticness, speechiness e instrumentalness?")

# Calcular o 75º percentil de popularidade para identificar músicas populares
popularity_values = df.select('popularity').rdd.flatMap(lambda x: x).collect()
if len(popularity_values) > 0:
    popularity_75th_percentile = df.approxQuantile('popularity', [0.75], 0.01)[0]
    st.write(f"75º Percentil da Popularidade: {popularity_75th_percentile}")

    # Filtrar as músicas populares
    popular_songs = df.filter(col('popularity') > popularity_75th_percentile)

    # Descrever as características das músicas populares
    stats = popular_songs.select('acousticness', 'speechiness', 'instrumentalness').describe().toPandas()
    st.write(stats)

    # Visualizar as características com boxplots
    fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(y=popular_songs.select('acousticness').toPandas()['acousticness'], ax=axes6[0]).set_title('Acousticness')
    sns.boxplot(y=popular_songs.select('speechiness').toPandas()['speechiness'], ax=axes6[1]).set_title('Speechiness')
    sns.boxplot(y=popular_songs.select('instrumentalness').toPandas()['instrumentalness'], ax=axes6[2]).set_title('Instrumentalness')
    fig6.suptitle('Características Comuns das Músicas Mais Populares')
    st.pyplot(fig6)
else:
    st.error("Não foi possível calcular o 75º percentil porque a coluna 'popularity' está vazia.")


# Filtrar as músicas populares
popular_songs = df.filter(col('popularity') > popularity_75th_percentile)

# Descrever as características das músicas populares
stats = popular_songs.select('acousticness', 'speechiness', 'instrumentalness').describe().toPandas()
st.write(stats)

# Visualizar as características com boxplots
fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(y=popular_songs.select('acousticness').toPandas()['acousticness'], ax=axes6[0]).set_title('Acousticness')
sns.boxplot(y=popular_songs.select('speechiness').toPandas()['speechiness'], ax=axes6[1]).set_title('Speechiness')
sns.boxplot(y=popular_songs.select('instrumentalness').toPandas()['instrumentalness'], ax=axes6[2]).set_title('Instrumentalness')
fig6.suptitle('Características Comuns das Músicas Mais Populares')
st.pyplot(fig6)

st.markdown("""
## Conclusão
### Quais são as características comuns das músicas mais populares em termos de acousticness, speechiness e instrumentalness?

As músicas mais populares tendem a ter valores moderados de acousticness e speechiness, enquanto a instrumentalness é geralmente baixa. Isso sugere que músicas com uma combinação equilibrada de elementos acústicos e vocais são mais populares.
""")

# Pergunta 7: Qual é a distribuição de popularidade por ano?
st.markdown("## Pergunta 7: Qual é a distribuição de popularidade por ano?")
df_popularity_year = df.filter((col('year') >= 2000) & (col('year') <= 2023))
fig7, ax7 = plt.subplots(figsize=(14, 10))
sns.boxplot(data=df_popularity_year.toPandas(), x='year', y='popularity', ax=ax7)
ax7.set_title('Distribuição de Popularidade por Ano (2000-2023)')
ax7.set_xlabel('Ano')
ax7.set_ylabel('Popularidade')
st.pyplot(fig7)

st.markdown("""
## Conclusão
### Qual é a distribuição de popularidade por ano?

A popularidade das músicas tem aumentado consistentemente ao longo dos anos, com uma maior diversidade de músicas populares em anos mais recentes.
""")

# Pergunta 8: Quais são os tempos de assinatura mais comuns em músicas populares?
st.markdown("## Pergunta 8: Quais são os tempos de assinatura mais comuns em músicas populares?")
fig8, ax8 = plt.subplots(figsize=(12, 6))
sns.countplot(data=popular_songs.toPandas(), x='time_signature', ax=ax8)
ax8.set_title('Tempos de Assinatura Mais Comuns em Músicas Populares')
ax8.set_xlabel('Time Signature')
ax8.set_ylabel('Count')
st.pyplot(fig8)

st.markdown("""
## Conclusão
### Quais são os tempos de assinatura mais comuns em músicas populares?

A maioria das músicas populares tem uma assinatura de tempo de 4/4, que é o tempo mais comum na música popular. Outros tempos de assinatura são muito menos frequentes.
""")

# Pergunta 9: Existe uma correlação entre o tempo de lançamento da música e sua popularidade futura?
st.markdown("## Pergunta 9: Existe uma correlação entre o tempo de lançamento da música e sua popularidade futura?")
df_popularity_year = df.filter((col('year') >= 2000) & (col('year') <= 2023))
correlation_year_popularity = df_popularity_year.corr('year', 'popularity')
st.write(f'Correlação entre o ano de lançamento e a popularidade: {correlation_year_popularity}')

fig9, ax9 = plt.subplots(figsize=(14, 10))
sns.scatterplot(data=df_popularity_year.toPandas(), x='year', y='popularity', alpha=0.5, ax=ax9, label='Músicas Individuais')
sns.lineplot(data=df_popularity_year.groupBy('year').avg('popularity').orderBy('year').toPandas(), x='year', y='avg(popularity)', color='red', ax=ax9, label='Média Anual')
ax9.set_title(f'Correlação entre Ano de Lançamento e Popularidade das Músicas (2000-2023)\nCorrelação: {correlation_year_popularity}')
ax9.set_xlabel('Ano de Lançamento')
ax9.set_ylabel('Popularidade')
ax9.legend()
st.pyplot(fig9)

st.markdown("""
## Conclusão
### Existe uma correlação entre o tempo de lançamento da música e sua popularidade futura?

A correlação entre o ano de lançamento e a popularidade é positiva, embora moderada. Isso sugere que músicas lançadas mais recentemente tendem a ser ligeiramente mais populares.
""")

# Pergunta 10: Qual a correlação entre loudness e popularidade das músicas?
st.markdown("## Pergunta 10: Qual a correlação entre loudness e popularidade das músicas?")
correlation_loudness = df.corr('loudness', 'popularity')
st.write(f'Correlação entre Loudness e Popularidade: {correlation_loudness}')

fig10, ax10 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='loudness', y='popularity', data=sample_df, alpha=0.7, ax=ax10)
sns.regplot(x='loudness', y='popularity', data=sample_df, scatter=False, color='red', ax=ax10)
ax10.set_xlabel('Loudness (dB)')
ax10.set_ylabel('Popularidade')
ax10.set_title('Correlação entre Loudness e Popularidade')
st.pyplot(fig10)

st.markdown("""
## Conclusão
### Qual a correlação entre loudness e popularidade das músicas?

A correlação entre loudness e popularidade é fraca, sugerindo que a intensidade sonora de uma música não tem um impacto significativo na sua popularidade.
""")
