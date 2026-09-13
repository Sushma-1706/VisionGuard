from app.grounding.verifier import iou, verify_claims
from app.schemas.inspection import BoundingBox, Claim
def test_matching_claims_are_grounded():
    result=verify_claims([Claim(type="defect",value="fracture"),Claim(type="location",value="upper-right")],"crack",.9,BoundingBox(x1=60,y1=0,x2=100,y2=40),100,100)
    assert result.supported_claims == 2 and result.score > .5
def test_unsupported_defect_is_detected():
    result=verify_claims([Claim(type="defect",value="rust")],"crack",.9,None,100,100)
    assert result.verifications[0].status == "unsupported"
def test_iou(): assert iou(BoundingBox(x1=0,y1=0,x2=10,y2=10),BoundingBox(x1=5,y1=5,x2=10,y2=10)) == .25
def test_iou_handles_degenerate_boxes(): assert iou(BoundingBox(x1=0,y1=0,x2=0,y2=0),BoundingBox(x1=0,y1=0,x2=0,y2=0)) == 0
def test_small_box_in_claimed_region_is_supported():
    result=verify_claims([Claim(type="location",value="upper-right")],"crack",.9,BoundingBox(x1=90,y1=0,x2=100,y2=10),100,100)
    assert result.verifications[0].status == "supported"
def test_no_claims_is_safe():
    assert verify_claims([],"crack",.9,None,100,100).score == 0
