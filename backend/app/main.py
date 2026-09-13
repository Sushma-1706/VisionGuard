import json, logging, uuid
from datetime import datetime, timezone
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .database import Inspection, get_inspection, init_database, save_inspection
from .ml.model import CLASS_NAMES
from .ml.preprocessing import InvalidImageError, load_image, validate_filename_extension
from .schemas.inspection import InspectionResponse, ModelInfo
from .services.inspection import inspection_service

logging.basicConfig(level=logging.INFO); log=logging.getLogger(__name__)
app=FastAPI(title="VisionGuard",version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins.split(","), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
@app.on_event("startup")
def startup() -> None: init_database()
@app.get("/api/v1/health")
def health(): return {"status":"ok","model_ready":inspection_service.ready}
@app.get("/api/v1/model-info",response_model=ModelInfo)
def model_info():
    return ModelInfo(name="ResNet-18 Grad-CAM",version="0.1.0",training_dataset="NEU Surface Defect Database",supported_defect_categories=CLASS_NAMES,metrics=inspection_service.metadata.get("metrics"),checkpoint_loaded=inspection_service.ready)
@app.post("/api/v1/inspect",response_model=InspectionResponse)
async def inspect(image: UploadFile=File(...)):
    try:
        source=load_image(await image.read(), settings.max_upload_bytes, settings.max_image_pixels, settings.max_image_dimension)
        validate_filename_extension(image.filename, source)
    except InvalidImageError as exc: raise HTTPException(422,str(exc)) from exc
    if not inspection_service.ready: raise HTTPException(503,"No trained model checkpoint is available. Train one with ml/training/train.py before inspecting images.")
    try: defect,confidence,localization,anomaly,explanation,grounding,uncertainty=inspection_service.inspect(source)
    except Exception as exc:
        log.exception("Inspection inference failed"); raise HTTPException(500,"Inspection could not be completed.") from exc
    response=InspectionResponse(inspection_id=str(uuid.uuid4()),status="normal" if defect=="normal" else "defective",defect_type=defect,confidence=confidence,localization=localization,anomaly_score=anomaly,explanation=explanation,grounding=grounding,uncertainty=uncertainty,created_at=datetime.now(timezone.utc))
    try: save_inspection(Inspection(id=response.inspection_id,filename=image.filename or "upload",prediction=response.status,defect_type=defect,confidence=confidence,anomaly_score=anomaly,grounding_score=grounding.score,hallucination_risk=grounding.hallucination_risk,explanation=explanation.description,result_json=response.model_dump_json()))
    except Exception: log.exception("Could not save inspection history")
    return response
@app.get("/api/v1/inspection/{inspection_id}",response_model=InspectionResponse)
def inspection(inspection_id: str):
    result=get_inspection(inspection_id)
    if not result: raise HTTPException(404,"Inspection not found.")
    return result
