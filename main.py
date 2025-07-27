import requests
import json
import PyPDF2
import csv
import tempfile
import base64
import fitz  # PyMuPDF
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from PIL import Image
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("API Key found:", bool(GEMINI_API_KEY))

app = Flask(__name__)

def get_gemini_api_key():
    # Use the global variable loaded at startup
    print(f"Looking for GEMINI_API_KEY in environment...")
    print(f"Available Gemini-related env vars: {[k for k in os.environ.keys() if 'GEMINI' in k.upper()]}")
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not found")
        print("Make sure your .env file contains: GEMINI_API_KEY=your_api_key_here")
        return None
    print(f"Found API key: {GEMINI_API_KEY[:10]}...")
    return GEMINI_API_KEY

def pdf_to_images(pdf_file, dpi=300):
    """Convert PDF pages to images using PyMuPDF - FIRST PAGE ONLY"""
    try:
        print("Attempting to convert PDF to images using PyMuPDF (first page only)...")
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            pdf_file.save(temp_pdf.name)
            temp_pdf_path = temp_pdf.name
        
        try:
            doc = fitz.open(temp_pdf_path)
            print(f"PDF opened successfully, has {len(doc)} pages, converting first page only")
            
            if len(doc) == 0:
                return None
            
            # Only convert the first page
            page = doc[0]
            try:
                print(f"Converting first page to image...")
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images = [img]  # Only return the first page image
                print(f"Successfully converted first page")
            except Exception as e:
                print(f"Error converting first page: {e}")
                return None
            
            print(f"Converted PDF first page to image")
            return images
        finally:
            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
                
    except Exception as e:
        print(f"PDF to image conversion failed: {e}")
        return None

def encode_image(image):
    """Encode PIL image as Base64."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def gemini_extract_text(images):
    """Extract text from images using Gemini Vision API - FIRST PAGE ONLY"""
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please set GEMINI_API_KEY environment variable."
    
    if not images or len(images) == 0:
        return "No images to process."
    
    print("Beginning OCR text extraction via Gemini Vision API (first page only)...")
    
    # Only process the first image/page
    image = images[0]
    try:
        print(f"Processing first page image...")
        base64_image = encode_image(image)
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{
                "parts": [
                    {"text": "Extract all text from this image."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }]
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        
        page_text = result["candidates"][0]["content"]["parts"][0]["text"]
        extracted_text = f"Page 1:\n{page_text}\n\n"
        print(f"Successfully extracted {len(page_text)} characters from first page")
        
        return extracted_text.strip()
        
    except Exception as e:
        print(f"Error processing first page image: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            return f"Gemini API quota exceeded. Please check your billing or try again later. Error: {str(e)}"
        return f"Error processing first page: {str(e)}"

def extract_text_with_pypdf2(pdf_file):
    """Fallback: Extract text using PyPDF2 - FIRST PAGE ONLY"""
    try:
        print("Attempting text extraction with PyPDF2 (first page only)...")
        reader = PyPDF2.PdfReader(pdf_file)
        print(f"PDF has {len(reader.pages)} pages, extracting from first page only")
        
        if len(reader.pages) == 0:
            return None
        
        # Only extract from first page
        page = reader.pages[0]
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text = f"Page 1:\n{page_text.strip()}\n\n"
                print(f"Extracted {len(page_text.strip())} characters from first page")
                return text
            else:
                print("No text found on first page")
                return None
        except Exception as e:
            print(f"Error extracting text from first page: {e}")
            return None
        
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
        return None

def extract_text_from_pdf_working(pdf_file):
    """Extract text from PDF using multiple approaches"""
    try:
        # First try: Use PyPDF2 for direct text extraction (no API calls needed)
        print("Method 1: Direct text extraction with PyPDF2...")
        text = extract_text_with_pypdf2(pdf_file)
        
        if text and text.strip():
            print(f"PyPDF2 extraction successful, length: {len(text)}")
            return text
        
        # Second try: Convert PDF to images and use Gemini Vision (if API quota available)
        print("Method 1 failed, trying Method 2: Converting PDF to images...")
        images = pdf_to_images(pdf_file)
        
        if images and len(images) > 0:
            print("Method 2 successful, extracting text using Gemini Vision API...")
            text = gemini_extract_text(images)
            if text and not text.startswith("Gemini API key not found") and not text.startswith("Gemini API quota exceeded"):
                print(f"Gemini Vision extraction successful, length: {len(text)}")
                return text
            elif text.startswith("Gemini API quota exceeded"):
                print("Gemini quota exceeded, falling back to PyPDF2 only")
                return "Gemini API quota exceeded. Text extraction completed using PyPDF2 only."
        
        return "Failed to extract text from PDF using available methods."
        
    except Exception as e:
        print(f"PDF processing failed: {e}")
        return f"PDF processing failed: {str(e)}"

def analyze_document_with_gemini(api_key, document_text):
    """Analyze document using Gemini API to extract patient information"""
    # Ensure document_text is a string
    if document_text is None:
        document_text = ""
    elif not isinstance(document_text, str):
        document_text = str(document_text)
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        prompt = (
            "You are a medical AI assistant. Extract the following fields from the provided medical document. "
            "Return each field as 'Field: Value' on its own line. If a field is missing, say 'Not specified'.\n\n"
            "Fields to extract:\n"
            "Patient Last Name\n"
            "Patient First Name\n"
            "Patient Middle Initial\n"
            "Patient Date of Birth\n"
            "Patient Gender\n"
            "Patient Driver's Licence Number\n"
            "Patient Unit Number\n"
            "Patient Street Number\n"
            "Patient Street Name\n"
            "Patient PO Box\n"
            "Patient City/Town/Village\n"
            "Patient Province\n"
            "Patient Postal Code\n"
            "Practitioner Last Name\n"
            "Practitioner First Name\n"
            "Practitioner Licence Number\n"
            "Practitioner Telephone Number\n"
            "Practitioner Unit Number\n"
            "Practitioner Street Number\n"
            "Practitioner Street Name\n"
            "Practitioner City/Town/Village\n"
            "Practitioner Province\n"
            "Practitioner Postal Code\n"
            "Practitioner Role\n"
            "Patient is aware of this report\n"
            "Notify if patient requests copy\n"
            "Practitioner's Signature\n"
            "Date of Report Examination\n"
            "Cognitive Impairment\n"
            "Cognitive Impairment Due To\n"
            "Sudden Incapacitation\n"
            "Sudden Incapacitation Due To\n"
            "Seizure\n"
            "Seizure Due To\n"
            "Syncope\n"
            "Syncope Due To\n"
            "CVA resulting in (check all that apply)\n"
            "CVA Due To\n"
            "Motor or Sensory Impairment\n"
            "Motor or Sensory Impairment Due To\n"
            "Visual Impairment\n"
            "Visual Impairment Details\n"
            "Substance Use Disorder\n"
            "Substance Use Disorder Details\n"
            "Psychiatric Disorder\n"
            "Psychiatric Disorder Details\n"
            "Other (specify)\n"
            "Discretionary Report of Medical Condition or Impairment\n"
            "Describe condition(s) or impairment\n"
            "\nMedical Document:\n" + document_text
        )
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        
        return result["candidates"][0]["content"]["parts"][0]["text"]
        
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return f"Gemini API quota exceeded. Cannot analyze document. Error: {str(e)}"
        else:
            return f"Gemini API error: {str(e)}"

def parse_ai_response(ai_text):
    fields = [
        "Patient Last Name", "Patient First Name", "Patient Middle Initial", "Patient Date of Birth",
        "Patient Gender", "Patient Driver's Licence Number", "Patient Unit Number", "Patient Street Number",
        "Patient Street Name", "Patient PO Box", "Patient City/Town/Village", "Patient Province",
        "Patient Postal Code", "Practitioner Last Name", "Practitioner First Name", "Practitioner Licence Number",
        "Practitioner Telephone Number", "Practitioner Unit Number", "Practitioner Street Number",
        "Practitioner Street Name", "Practitioner City/Town/Village", "Practitioner Province",
        "Practitioner Postal Code", "Practitioner Role", "Patient is aware of this report",
        "Notify if patient requests copy", "Practitioner's Signature", "Date of Report Examination",
        "Cognitive Impairment", "Cognitive Impairment Due To", "Sudden Incapacitation", "Sudden Incapacitation Due To",
        "Seizure", "Seizure Due To", "Syncope", "Syncope Due To",
        "CVA resulting in (check all that apply)", "CVA Due To",
        "Motor or Sensory Impairment", "Motor or Sensory Impairment Due To",
        "Visual Impairment", "Visual Impairment Details",
        "Substance Use Disorder", "Substance Use Disorder Details",
        "Psychiatric Disorder", "Psychiatric Disorder Details",
        "Other (specify)", "Discretionary Report of Medical Condition or Impairment",
        "Describe condition(s) or impairment"
    ]
    result = []
    for field in fields:
        value = "Not specified"
        for line in ai_text.splitlines():
            if line.lower().startswith(field.lower()):
                value = line.split(":", 1)[1].strip()
                break
        result.append(value)
    return result

def append_to_csv(row, filename="extracted_info.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Patient Last Name", "Patient First Name", "Patient Middle Initial", "Patient Date of Birth",
                "Patient Gender", "Patient Driver's Licence Number", "Patient Unit Number", "Patient Street Number",
                "Patient Street Name", "Patient PO Box", "Patient City/Town/Village", "Patient Province",
                "Patient Postal Code", "Practitioner Last Name", "Practitioner First Name", "Practitioner Licence Number",
                "Practitioner Telephone Number", "Practitioner Unit Number", "Practitioner Street Number",
                "Practitioner Street Name", "Practitioner City/Town/Village", "Practitioner Province",
                "Practitioner Postal Code", "Practitioner Role", "Patient is aware of this report",
                "Notify if patient requests copy", "Practitioner's Signature", "Date of Report Examination",
                "Cognitive Impairment", "Cognitive Impairment Due To", "Sudden Incapacitation", "Sudden Incapacitation Due To",
                "Seizure", "Seizure Due To", "Syncope", "Syncope Due To",
                "CVA resulting in (check all that apply)", "CVA Due To",
                "Motor or Sensory Impairment", "Motor or Sensory Impairment Due To",
                "Visual Impairment", "Visual Impairment Details",
                "Substance Use Disorder", "Substance Use Disorder Details",
                "Psychiatric Disorder", "Psychiatric Disorder Details",
                "Other (specify)", "Discretionary Report of Medical Condition or Impairment",
                "Describe condition(s) or impairment"
            ])
        writer.writerow(row)

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    api_key = get_gemini_api_key()
    file = request.files.get('pdf_file')
    if not file:
        return jsonify({'success': False, 'error': 'PDF file is required.'})
    
    try:
        # Extract text using the working approach
        text = extract_text_from_pdf_working(file)
        
        if not text or text.startswith("Failed to extract"):
            return jsonify({'success': False, 'error': text})
        
        if not text.strip():
            return jsonify({'success': False, 'error': 'No text could be extracted from the PDF.'})
        
        print(f"Text type: {type(text)}, length: {len(text) if text else 0}")
        print(f"Text preview: {text[:200] if text else 'None'}")
        
        # Check if we have Gemini API available for analysis
        if api_key and not text.startswith("Gemini API quota exceeded"):
            # Analyze the extracted text with Gemini
            print("Calling Gemini for analysis...")
            result = analyze_document_with_gemini(api_key, text)
            print(f"Gemini result type: {type(result)}, length: {len(result) if result else 0}")
            print(f"Gemini result preview: {result[:200] if result else 'None'}")
            
            if result and not result.startswith("Gemini API quota exceeded") and not result.startswith("Gemini API error"):
                # Parse the AI response and save to CSV
                print("Parsing AI response...")
                row = parse_ai_response(result)
                print(f"Parsed row length: {len(row)}")
                print(f"Parsed row preview: {row[:5] if row else 'None'}")
                
                print("Saving to CSV...")
                append_to_csv(row)
                print("CSV saved successfully!")
                
                return jsonify({'success': True, 'result': result})
            else:
                # Gemini analysis failed, return extracted text only
                print(f"Gemini analysis failed: {result}")
                return jsonify({
                    'success': True, 
                    'result': f"Text extracted successfully but Gemini analysis failed.\n\nExtracted Text:\n{text[:1000]}...",
                    'warning': 'Gemini analysis failed. Only text extraction completed.'
                })
        else:
            # No Gemini API available, return extracted text only
            print("No Gemini API available for analysis")
            return jsonify({
                'success': True, 
                'result': f"Text extracted successfully using PyPDF2.\n\nExtracted Text:\n{text[:1000]}...",
                'warning': 'No Gemini API available. Only text extraction completed.'
            })
        
    except Exception as e:
        print(f"Error in analyze_pdf: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Failed to extract text from PDF: {str(e)}'})

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
