from io import BytesIO
from PIL import Image
import pytest
from app.ml.preprocessing import InvalidImageError, load_image, validate_filename_extension
def test_valid_png_loads():
    b=BytesIO();Image.new("RGB",(32,32)).save(b,"PNG");assert load_image(b.getvalue(),10000).mode=="RGB"
def test_corrupt_image_rejected():
    with pytest.raises(InvalidImageError): load_image(b"nope",10000)
def test_extension_must_match_decoded_format():
    b=BytesIO();Image.new("RGB",(32,32)).save(b,"PNG")
    with pytest.raises(InvalidImageError): validate_filename_extension("surface.jpg",load_image(b.getvalue(),10000))
def test_pixel_limit_is_enforced():
    b=BytesIO();Image.new("RGB",(32,32)).save(b,"PNG")
    with pytest.raises(InvalidImageError): load_image(b.getvalue(),10000,max_pixels=100)
