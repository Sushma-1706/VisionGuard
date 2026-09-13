from io import BytesIO
from PIL import Image
import pytest
from app.ml.preprocessing import InvalidImageError, load_image
def test_valid_png_loads():
    b=BytesIO();Image.new("RGB",(32,32)).save(b,"PNG");assert load_image(b.getvalue(),10000).mode=="RGB"
def test_corrupt_image_rejected():
    with pytest.raises(InvalidImageError): load_image(b"nope",10000)
