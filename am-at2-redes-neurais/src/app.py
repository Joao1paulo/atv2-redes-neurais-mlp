from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.schemas import PacienteInput, PredicaoOutput

BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = BASE_DIR / "models" / "pipeline_rede_neural.joblib"

MAPA_CLASSES = {
    0: "Normal",
    1: "Risco Cardíaco",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da API carregando o pipeline apenas na inicialização."""
    if not CAMINHO_MODELO.exists():
        raise RuntimeError(
            f"Artefato do modelo não encontrado em '{CAMINHO_MODELO}'. "
            "Execute o notebook notebooks/analise_treinamento.ipynb antes de iniciar a API."
        )

    app.state.modelo = joblib.load(CAMINHO_MODELO)
    yield
    app.state.modelo = None


app = FastAPI(
    title="API de Triagem Cardiovascular - MLPClassifier",
    description="Microsserviço RESTful para inferência de risco cardíaco utilizando Pipeline Scikit-Learn.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/diagnostico/modelo", status_code=200)
async def diagnostico_modelo() -> dict[str, str]:
    """Retorna os metadados e o status operacional do serviço de inferência."""
    return {
        "status": "operacional_v2",
        "framework": "FastAPI-Production",
        "modelo": "MLPClassifier",
    }


@app.post("/predict", response_model=PredicaoOutput, status_code=200)
async def prever_risco_cardiaco(
    dados: PacienteInput, request: Request
) -> PredicaoOutput:
    """Recebe os atributos clínicos validados e executa a inferência com o Pipeline carregado."""
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo preditivo não está carregado na memória da aplicação.",
        )

    entrada_df = pd.DataFrame([dados.model_dump()])

    classe_numerica = int(modelo.predict(entrada_df)[0])
    probabilidades = modelo.predict_proba(entrada_df)[0]
    confianca = round(float(np.max(probabilidades)), 4)

    categoria = MAPA_CLASSES.get(classe_numerica, str(classe_numerica))

    return PredicaoOutput(
        categoria_prevista=categoria,
        score_confianca=confianca,
    )