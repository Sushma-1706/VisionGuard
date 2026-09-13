from io import BytesIO
from PIL import Image, UnidentifiedImageError
import torch
from torchvision.transforms import v2

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
class InvalidImageError(ValueError): pass
def load_image(data: bytes, max_bytes: int) -> Image.Image:
    if not data or len(data) > max_bytes: raise InvalidImageError("Image is empty or exceeds the 10 MB upload limit.")
    try:
        with Image.open(BytesIO(data)) as source:
            source.verify()
        image = Image.open(BytesIO(data))
        if image.format not in SUPPORTED_FORMATS: raise InvalidImageError("Supported formats: JPG, JPEG, PNG, WEBP.")
        if image.width < 16 or image.height < 16: raise InvalidImageError("Image dimensions must be at least 16 × 16 pixels.")
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc: raise InvalidImageError("The uploaded file is corrupted or not a valid image.") from exc
def image_transform(size: int) -> v2.Compose:
    return v2.Compose([v2.Resize((size, size)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
