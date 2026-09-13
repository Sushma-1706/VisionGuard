from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
from torchvision.transforms import v2

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
FORMAT_EXTENSIONS = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "WEBP": {".webp"}}
class InvalidImageError(ValueError): pass
def load_image(data: bytes, max_bytes: int, max_pixels: int = 4_000_000, max_dimension: int = 4_096) -> Image.Image:
    if not data or len(data) > max_bytes: raise InvalidImageError("Image is empty or exceeds the 10 MB upload limit.")
    try:
        with Image.open(BytesIO(data)) as source:
            source.verify()
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS: raise InvalidImageError("Supported formats: JPG, JPEG, PNG, WEBP.")
            if image.width < 16 or image.height < 16: raise InvalidImageError("Image dimensions must be at least 16 × 16 pixels.")
            if image.width > max_dimension or image.height > max_dimension or image.width * image.height > max_pixels:
                raise InvalidImageError("Decoded image dimensions exceed the safe processing limit.")
            converted = image.convert("RGB")
            converted.info["visionguard_format"] = image_format
            return converted
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc: raise InvalidImageError("The uploaded file is corrupted or not a valid image.") from exc

def validate_filename_extension(filename: str | None, image: Image.Image) -> None:
    """Require the filename extension, when supplied, to match decoded magic bytes."""
    if not filename:
        return
    extension = Path(filename).suffix.lower()
    detected_format = image.info.get("visionguard_format")
    if extension not in FORMAT_EXTENSIONS.get(detected_format, set()):
        raise InvalidImageError("Filename extension does not match the decoded image format.")
def image_transform(size: int) -> v2.Compose:
    return v2.Compose([v2.Resize((size, size)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
