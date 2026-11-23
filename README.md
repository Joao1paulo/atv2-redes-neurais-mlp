# Predição de Risco Cardiovascular com Rede Neural MLP e FastAPI

Projeto de aprendizado de máquina supervisionado desenvolvido com **Scikit-Learn (`MLPClassifier`)**, **FastAPI**, **Pydantic** e gerenciamento de ambiente via **UV**.

## Dataset Utilizado

* **Nome:** Heart Failure Prediction Dataset
* **Fonte (Kaggle):** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
* **Volume:** 918 registros e 11 variáveis preditoras (numéricas e categóricas) + 1 alvo binário (`HeartDisease`).

## Estrutura do Projeto

```text
am-at2-redes-neurais/
├── data/
│   └── dataset.csv                  # Base original obtida no Kaggle (heart.csv)
├── models/
│   └── pipeline_rede_neural.joblib  # Pipeline serializado (ColumnTransformer + MLPClassifier)
├── notebooks/
│   └── analise_treinamento.ipynb    # EDA, Pipeline, Métricas (MCC/Acurácia) e Exportação
├── src/
│   ├── __init__.py
│   ├── schemas.py                   # Contratos Pydantic de entrada e saída
│   └── app.py                       # Servidor FastAPI com Lifespan e endpoints
├── pyproject.toml                   # Dependências gerenciadas pelo UV
├── requisicoes.http                 # Testes de integração HTTP
└── README.md