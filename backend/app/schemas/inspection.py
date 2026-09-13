from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: int; y1: int; x2: int; y2: int

class Localization(BaseModel):
    bbox: BoundingBox | None = None
    region: str
    heatmap_png_base64: str | None = None
    overlay_png_base64: str | None = None

class Claim(BaseModel):
    type: Literal["defect", "location", "severity"]
    value: str

class Explanation(BaseModel):
    provider: str
    description: str
    severity: str
    evidence: str
    claims: list[Claim]

class ClaimVerification(BaseModel):
    claim: Claim
    status: Literal["supported", "weak", "unsupported"]
    evidence_available: bool
    semantic_consistency: float | None = None
    spatial_consistency: float | None = None
    evidence_confidence: float
    reason: str

class Grounding(BaseModel):
    score: float = Field(ge=0, le=1)
    supported_claims: int
    total_claims: int
    hallucination_risk: Literal["low", "medium", "high"]
    verifications: list[ClaimVerification]

class Uncertainty(BaseModel):
    model_confidence: float
    grounding_confidence: float
    evidence_strength: float
    level: Literal["low", "medium", "high"]
    note: str

class InspectionResponse(BaseModel):
    inspection_id: str
    status: Literal["normal", "defective"]
    defect_type: str
    confidence: float
    localization: Localization
    anomaly_score: float
    explanation: Explanation
    grounding: Grounding
    uncertainty: Uncertainty
    created_at: datetime

class ModelInfo(BaseModel):
    name: str; version: str; training_dataset: str
    supported_defect_categories: list[str]
    metrics: dict[str, float] | None
    checkpoint_loaded: bool
