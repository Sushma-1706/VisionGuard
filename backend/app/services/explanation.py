from ..schemas.inspection import Claim, Explanation

def evidence_narrator(defect: str, confidence: float, region: str, anomaly_score: float) -> Explanation:
    """A local evidence narrator, not a VLM: it only verbalizes supplied model evidence."""
    if defect == "normal":
        description="No trained defect category exceeded the decision threshold in the inspected surface."
        claims=[Claim(type="defect", value="normal")]
    else:
        readable=defect.replace("_", " ")
        description=f"The classifier predicts {readable} in the {region} region; the Grad-CAM evidence map has a normalized response of {anomaly_score:.2f}."
        claims=[Claim(type="defect",value=readable), Claim(type="location",value=region), Claim(type="severity",value="requires review")]
    return Explanation(provider="local-evidence-narrator",description=description,severity="requires review",evidence=f"Model confidence {confidence:.1%}; anomaly response {anomaly_score:.2f}.",claims=claims)
