#!./env/bin/python
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-29 23:57:01 (ywatanabe)"

# pptx2tif.py

import io

import cv2
from PIL import Image
from pptx import Presentation


def find_content_area(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read the image file: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, x + w, y + h


def convert_pptx_to_tif(pptx_path):
    pres = Presentation(pptx_path)
    for i, slide in enumerate(pres.slides):
        print()
        # Export each slide as an image
        image = Image.open(io.BytesIO(slide.shapes[0].image.blob))
        # Save the image as TIFF
        image.save(f"slide_{i}.tif", format="TIFF")  # [REVISED]


if __name__ == "__main__":

    pptx_path = "./src/figures/Figure_01.pptx"

    # import aspose.slides as slides

    # presentation = slides.Presentation(pptx_path)
    # presentation.save(
    #     pptx_path.replace(".pptx", ".tif"), slides.export.SaveFormat.TIFF
    # )
