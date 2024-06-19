# Utilize a imagem oficial do Jupyter Notebook base
FROM jupyter/base-notebook:latest

# Define variáveis de ambiente para o Spark
ENV SPARK_VERSION=3.4.3 \
    HADOOP_VERSION=3 \
    SPARK_HOME=/usr/local/spark \
    PATH=$SPARK_HOME/bin:$PATH

# Altera para o usuário root para instalar as dependências
USER root

# Instale as dependências necessárias e baixe/instale o Spark
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jdk wget && \
    wget --no-check-certificate https://downloads.apache.org/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz -O /tmp/spark.tgz && \
    tar xzvf /tmp/spark.tgz -C /usr/local/ && \
    mv /usr/local/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION $SPARK_HOME && \
    rm /tmp/spark.tgz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Instale as bibliotecas Python necessárias
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pymongo pandas numpy matplotlib scikit-learn seaborn streamlit streamlit_jupyter redis plotly ipywidgets pyspark jupyter_contrib_nbextensions statsmodels
#pyspark==${SPARK_VERSION}

# Instale e habilite nbformat e outras dependências Jupyter
RUN pip install ipykernel && \
    pip install --upgrade nbformat notebook jupyter-client jupyter-core


# Install and enable Jupyter nbextensions and JupyterLab extensions
RUN pip install notebook==6.5.4 && \
    jupyter contrib nbextension install --sys-prefix && \
    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    jupyter nbextension enable --py --sys-prefix jupyter_nbextensions_configurator && \
    jupyter nbextensions_configurator enable --sys-prefix && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    jupyter lab build --dev-build=False --minimize=False

RUN echo "jovyan ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/jovyan

# Troque para o usuário padrão do Jupyter Docker
USER jovyan

# Configure o diretório de trabalho padrão para os notebooks
WORKDIR /home/jovyan/work

