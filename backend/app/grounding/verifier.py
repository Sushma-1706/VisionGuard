from ..schemas.inspection import BoundingBox, Claim, ClaimVerification, Grounding

SYNONYMS = {"crack": {"crack", "fracture", "fissure"}, "scratches": {"scratch", "scratches", "scuff"}, "pitted_surface": {"pit", "pitted", "pitting"}, "rolled-in_scale": {"scale", "rolled-in scale"}, "inclusion": {"inclusion", "embedded particle"}, "patches": {"patch", "patches"}, "normal": {"normal", "no defect", "clean"}}
REGIONS = {"upper-left":(0,0,.5,.5), "upper-right":(.5,0,1,.5), "center":(.25,.25,.75,.75), "bottom-left":(0,.5,.5,1), "bottom-right":(.5,.5,1,1)}
def canonical_defect(text: str) -> str | None:
    normalized = text.lower().replace("_", " ")
    return next((label for label, words in SYNONYMS.items() if any(word in normalized for word in words)), None)
def location_box(region: str, width: int, height: int) -> BoundingBox | None:
    values = REGIONS.get(region.lower().replace(" region", ""));
    return BoundingBox(x1=int(values[0]*width),y1=int(values[1]*height),x2=int(values[2]*width),y2=int(values[3]*height)) if values else None
def iou(a: BoundingBox, b: BoundingBox) -> float:
    if a.x2 <= a.x1 or a.y2 <= a.y1 or b.x2 <= b.x1 or b.y2 <= b.y1:
        return 0.0
    left, top, right, bottom = max(a.x1,b.x1),max(a.y1,b.y1),min(a.x2,b.x2),min(a.y2,b.y2)
    inter=max(0,right-left)*max(0,bottom-top); union=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter
    return inter / union if union > 0 else 0.0

def box_center_in_region(region: BoundingBox, evidence: BoundingBox) -> bool:
    """Check the evidence-box center against a named region.

    Region language such as "upper-right" refers to where a defect lies, not
    to the area ratio covered by a defect. This avoids penalising small defects.
    """
    center_x = (evidence.x1 + evidence.x2) / 2
    center_y = (evidence.y1 + evidence.y2) / 2
    return region.x1 <= center_x <= region.x2 and region.y1 <= center_y <= region.y2
def verify_claims(claims: list[Claim], predicted: str, confidence: float, bbox: BoundingBox | None, width: int, height: int) -> Grounding:
    if not claims:
        return Grounding(score=0.0, supported_claims=0, total_claims=0, hallucination_risk="high", verifications=[])
    checks=[]
    for claim in claims:
        if claim.type == "defect":
            match=canonical_defect(claim.value)==predicted; score=confidence if match else 0.0
            checks.append(ClaimVerification(claim=claim,status="supported" if match else "unsupported",evidence_available=predicted!="normal",semantic_consistency=float(match),evidence_confidence=score,reason="Defect taxonomy agrees with the trained classifier." if match else f"Model evidence is '{predicted}', not '{claim.value}'."))
        elif claim.type == "location":
            claimed=location_box(claim.value,width,height)
            overlap=iou(claimed,bbox) if claimed and bbox else 0.0
            center_matches=bool(claimed and bbox and box_center_in_region(claimed,bbox))
            status="supported" if center_matches else "unsupported"
            checks.append(ClaimVerification(claim=claim,status=status,evidence_available=bbox is not None,spatial_consistency=float(center_matches),evidence_confidence=confidence if center_matches else 0.0,reason=f"Evidence-box center {'is' if center_matches else 'is not'} in the claimed region (region IoU: {overlap:.2f})."))
        else:
            # Severity has no direct target in this classifier; it is deliberately weak, not invented.
            checks.append(ClaimVerification(claim=claim,status="weak",evidence_available=False,evidence_confidence=0.0,reason="Severity is not an independently trained output in this baseline."))
    score_checks=[check for check in checks if check.claim.type != "severity"]
    score=sum(check.evidence_confidence for check in score_checks) / len(score_checks) if score_checks else 0.0
    risk="low" if score>=.75 and all(c.status=="supported" for c in score_checks) else "medium" if score>=.4 else "high"
    return Grounding(score=score,supported_claims=sum(c.status=="supported" for c in checks),total_claims=len(checks),hallucination_risk=risk,verifications=checks)
