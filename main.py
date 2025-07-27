import json
import csv
import tempfile
import os
import fitz  # PyMuPDF for PDF to image
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import re
from tabulate import tabulate  # ✅ For nice table output in terminal

# ✅ Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# ✅ Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Flask app
app = Flask(__name__)

# Fields for CSV and JSON
FIELDS = [
    "From","Patient Last Name", "Patient First Name", "Patient Middle Initial", "Patient Date of Birth",
    "Patient Gender", "Patient Driver's Licence Number", "Patient Unit Number", "Patient Street Number",
    "Patient Street Name", "Patient PO Box", "Patient City/Town/Village", "Patient Province",
    "Patient Postal Code", "Practitioner Last Name", "Practitioner First Name", "Practitioner Licence Number",
    "Practitioner Telephone Number", "Practitioner Unit Number", "Practitioner Street Number",
    "Practitioner Street Name", "Practitioner City/Town/Village", "Practitioner Province",
    "Practitioner Postal Code", "Practitioner Role", "Patient is aware of this report",
    "Notify if patient requests copy", "Practitioner's Signature", "Date of Report Examination"
]

# ✅ Convert PDF to image (first page)
def pdf_to_image_bytes(pdf_file, dpi=300):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        pdf_file.save(temp_pdf.name)
        temp_pdf_path = temp_pdf.name

    try:
        doc = fitz.open(temp_pdf_path)
        if len(doc) == 0:
            return None

        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()  # ✅ Close before deleting

        # Convert image to Base64
        buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(buffer.name, format="JPEG")

        with open(buffer.name, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        return encoded_image
    finally:
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

# ✅ Analyze image using Gemini Vision API
def analyze_pdf_image_with_gemini(encoded_image):
    model = genai.GenerativeModel("gemini-1.5-pro") 

    prompt = f"""
    Extract the following fields from this medical form image and return ONLY valid JSON with these exact keys:
    {", ".join(FIELDS)}
    If a value is missing, use "Not specified".
    Return ONLY JSON, no extra text.
    """

    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": base64.b64decode(encoded_image)}
    ])
    return response.text

# ✅ Clean Gemini Output
def clean_gemini_output(output):
    # Remove markdown code block markers like ```json and ```
    cleaned = re.sub(r"```[a-zA-Z]*", "", output).replace("```", "").strip()
    return cleaned

# ✅ Parse fallback text if JSON fails
def parse_text_response(text_response):
    data = {}
    lines = text_response.split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                for field in FIELDS:
                    if field.lower() in key.lower() or key.lower() in field.lower():
                        data[field] = value
                        break
    return data

# ✅ Save to CSV
def append_to_csv(data_dict, filename="extracted_info.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)
        f.flush()  # <-- This line forces data to disk

@app.route("/analyze_pdf", methods=["POST"])
def analyze_pdf():
    file = request.files.get("pdf_file")
    if not file:
        return jsonify({"success": False, "error": "PDF file is required."})

    try:
        # ✅ Convert PDF to image Base64
        encoded_image = pdf_to_image_bytes(file)
        if not encoded_image:
            return jsonify({"success": False, "error": "Failed to convert PDF to image."})

        # ✅ Analyze with Gemini Vision
        gemini_output_raw = analyze_pdf_image_with_gemini(encoded_image)
        print("\n=== Raw Gemini Output ===")
        print(gemini_output_raw)

        # ✅ Clean Gemini Output
        gemini_output = clean_gemini_output(gemini_output_raw)

        # ✅ Parse JSON or fallback
        try:
            data = json.loads(gemini_output)
        except json.JSONDecodeError:
            print("Gemini did not return valid JSON, using fallback parsing...")
            data = parse_text_response(gemini_output)

        # ✅ Fill missing fields
        for field in FIELDS:
            if field not in data:
                data[field] = "Not specified"

        # ✅ Print nicely to terminal
        print("\n=== Extracted Information ===")
        table = [(k, v) for k, v in data.items()]
        print(tabulate(table, headers=["Field", "Value"], tablefmt="grid"))

        # ✅ Save to CSV
        append_to_csv(data)

        # ✅ Save JSON backup
        with open("last_extracted.json", "w", encoding="utf-8") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f, indent=2)

        return jsonify({"success": True, "data": data})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    print("✅ Starting CloudAct Medical Document Analyzer...")
    app.run(debug=True)

