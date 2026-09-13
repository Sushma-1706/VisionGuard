import numpy as np, torch
from PIL import Image
from ..config import settings
from ..grounding.verifier import verify_claims
from ..ml.explainability import gradcam
from ..ml.model import CLASS_NAMES, load_checkpoint
from ..ml.preprocessing import image_transform
from ..schemas.inspection import BoundingBox, Localization, Uncertainty
from .explanation import evidence_narrator

class InspectionService:
    def __init__(self) -> None:
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); self._model=None; self.metadata={}
    @property
    def ready(self) -> bool: return settings.model_path.exists()
    def model(self):
        if self._model is None: self._model,self.metadata=load_checkpoint(settings.model_path,self.device)
        return self._model
    def inspect(self, image: Image.Image):
        tensor=image_transform(settings.image_size)(image).unsqueeze(0).to(self.device)
        model=self.model()
        with torch.no_grad(): probabilities=torch.softmax(model(tensor),dim=1)[0]
        predicted_index=int(probabilities.argmax()); confidence=float(probabilities[predicted_index]); defect=CLASS_NAMES[predicted_index]
        cam,heatmap,overlay=gradcam(model,tensor,predicted_index,image); threshold=max(.35, float(cam.mean()+cam.std()))
        ys,xs=np.where(cam>=threshold); bbox=None
        if defect!="normal" and len(xs): bbox=BoundingBox(x1=int(xs.min()),y1=int(ys.min()),x2=int(xs.max()+1),y2=int(ys.max()+1))
        center=(bbox.x1+bbox.x2)/2/image.width if bbox else .5; top=(bbox.y1+bbox.y2)/2/image.height if bbox else .5
        region=("upper" if top<.5 else "bottom")+("-left" if center<.5 else "-right") if abs(top-.5)>.16 and abs(center-.5)>.16 else "center"
        # The classifier's non-normal probability is a real, bounded anomaly
        # score. Do not derive it from a min-max normalized CAM (whose max is 1).
        anomaly=1.0 - float(probabilities[CLASS_NAMES.index("normal")])
        localization=Localization(bbox=bbox,region=region,heatmap_png_base64=heatmap,overlay_png_base64=overlay)
        explanation=evidence_narrator(defect,confidence,region,anomaly)
        grounding=verify_claims(explanation.claims,defect,confidence,bbox,image.width,image.height)
        level="low" if confidence>=.8 and grounding.score>=.7 else "medium" if confidence>=.55 else "high"
        uncertainty=Uncertainty(model_confidence=confidence,grounding_confidence=grounding.score,evidence_strength=anomaly,level=level,note="Confidence is the softmax probability from the trained classifier; it is not calibrated probability of correctness.")
        return defect,confidence,localization,anomaly,explanation,grounding,uncertainty

inspection_service=InspectionService()
