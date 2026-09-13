# Grounding method

VisionGuard treats text as a hypothesis. The `local-evidence-narrator` currently emits only claims derived from the classifier, Grad-CAM box, and anomaly response; a future VLM adapter must return the same structured schema and is still verified independently.

For defect claims, the verifier maps words through a controlled synonym taxonomy before comparing against the classifier label. For location claims, it turns one of five named regions into an image rectangle and computes IoU with the Grad-CAM evidence box. Severity is weak evidence because no severity label is trained. The grounding score is the mean of each claim's evidence confidence: matching defect claims receive model confidence, location claims receive `model_confidence × IoU`, and unsupported/unsupported-target claims receive zero. This formulation makes every score traceable rather than a generated quality estimate.
