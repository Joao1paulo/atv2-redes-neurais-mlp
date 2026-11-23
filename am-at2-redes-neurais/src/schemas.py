from pydantic import BaseModel, Field


class PacienteInput(BaseModel):
    """Contrato de entrada validado para predição de risco cardiovascular."""

    Age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Idade do paciente em anos completos",
        examples=[62],
    )
    Sex: str = Field(
        ...,
        pattern="^(M|F)$",
        description="Sexo biológico do paciente ('M' para Masculino, 'F' para Feminino)",
        examples=["M"],
    )
    ChestPainType: str = Field(
        ...,
        pattern="^(TA|ATA|NAP|ASY)$",
        description="Tipo de dor torácica ('TA', 'ATA', 'NAP' ou 'ASY')",
        examples=["ASY"],
    )
    RestingBP: int = Field(
        ...,
        ge=60,
        le=250,
        description="Pressão arterial em repouso (mm Hg)",
        examples=[145],
    )
    Cholesterol: int = Field(
        ...,
        ge=0,
        le=700,
        description="Colesterol sérico total (mg/dl)",
        examples=[289],
    )
    FastingBS: int = Field(
        ...,
        ge=0,
        le=1,
        description="Glicemia em jejum > 120 mg/dl (1 = Sim, 0 = Não)",
        examples=[1],
    )
    RestingECG: str = Field(
        ...,
        pattern="^(Normal|ST|LVH)$",
        description="Resultado do eletrocardiograma em repouso ('Normal', 'ST' ou 'LVH')",
        examples=["ST"],
    )
    MaxHR: int = Field(
        ...,
        ge=60,
        le=220,
        description="Frequência cardíaca máxima atingida (bpm)",
        examples=[112],
    )
    ExerciseAngina: str = Field(
        ...,
        pattern="^(Y|N)$",
        description="Angina induzida por exercício ('Y' = Sim, 'N' = Não)",
        examples=["Y"],
    )
    Oldpeak: float = Field(
        ...,
        ge=-5.0,
        le=10.0,
        description="Depressão do segmento ST induzida por exercício",
        examples=[2.5],
    )
    ST_Slope: str = Field(
        ...,
        pattern="^(Up|Flat|Down)$",
        description="Inclinação do segmento ST no pico do exercício ('Up', 'Flat' ou 'Down')",
        examples=["Flat"],
    )


class PredicaoOutput(BaseModel):
    """Contrato estruturado de saída retornado pelo endpoint de predição."""

    categoria_prevista: str = Field(
        ...,
        description="Rótulo textual da classe prevista pelo modelo",
        examples=["Risco Cardíaco"],
    )
    score_confianca: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiança da predição calculada via predict_proba (entre 0.0 e 1.0)",
        examples=[0.9345],
    )