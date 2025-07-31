import argparse
import cv2
import pytesseract
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Optional
import re

def image_to_numeric_dataframe(image_path: str, divide_by_1000: bool = False) -> pd.DataFrame:
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR configuration
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(binary, config=custom_config)

    # Process text into rows and columns
    rows = [line for line in extracted_text.split('\n') if line.strip()]
    table = [row.split() for row in rows]

    # Align rows to the same length
    max_len = max(len(row) for row in table)
    table = [row + [''] * (max_len - len(row)) for row in table]

    # Clean and convert to float
    def clean_to_float(value):
        # Keep digits, minus sign, and dot
        cleaned = re.sub(r'[^\d\.\-]', '', value)
        try:
            num = float(cleaned)
            return num / 1000 if divide_by_1000 else num
        except ValueError:
            return np.nan

    numeric_table = [[clean_to_float(cell) for cell in row] for row in table]

    df = pd.DataFrame(numeric_table)
    return df

# def main():
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#     image_path = "3.png"
#     out_path = "1.csv"
#     df= image_to_numeric_dataframe(image_path, True)
#     df.to_csv(out_path)
#     print(df.info())
#     print(df)

def main():
    parser = argparse.ArgumentParser(description="Extract numeric table from image and save as CSV.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("output_path", help="Path to save the CSV output")
    parser.add_argument("--divide", action="store_true", help="Divide all numbers by 1000")

    args = parser.parse_args()

    df = image_to_numeric_dataframe(args.image_path, divide_by_1000=args.divide)
    df.to_csv(args.output_path, index=False)
    print(f"Data saved to {args.output_path}")

if __name__=="__main__":
    main()