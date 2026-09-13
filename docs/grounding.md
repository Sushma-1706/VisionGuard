# Grounding method

VisionGuard treats text as a hypothesis. The `local-evidence-narrator` currently emits only claims derived from the classifier, Grad-CAM box, and anomaly response; a future VLM adapter must return the same structured schema and is still verified independently.

For defect claims, the verifier maps words through a controlled synonym taxonomy before comparing against the classifier label. For location claims, it turns one of five named regions into an image rectangle and verifies that the Grad-CAM evidence-box **center** lies inside it; the accompanying IoU is diagnostic only, because small defects should not fail a quadrant claim. Severity is weak evidence because no severity label is trained. The grounding score is the mean evidence confidence of *verifiable* defect and location claims; weak severity claims are reported but deliberately excluded. This formulation makes every score traceable rather than a generated quality estimate.
