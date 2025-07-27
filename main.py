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
import google.generativeai as genai

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
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-vision")
    
    # Only process the first image/page
    image = images[0]
    try:
        print(f"Processing first page image...")
        
        # Convert PIL image to bytes for Gemini
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        
        # Create the prompt
        prompt = "Extract all text from this image."
        
        # Generate content with image
        response = model.generate_content([prompt, image_bytes])
        
        page_text = response.text
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
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        prompt = """
You are a medical document extraction assistant. Extract the following fields from the provided medical document.
Return each field as 'Field: Value' on its own line. If a field is missing, say 'Not specified'.

IMPORTANT INSTRUCTIONS:
1. Look for any filled-in information in the document (especially blue text)
2. Extract any visible text, form fields, or filled information
3. Pay attention to any text that might be in different colors or formatting
4. Look for checkboxes that are marked, dates that are filled, or names that are written
5. Focus only on the fields that are actually present on this form

Fields to extract:
Patient Last Name
Patient First Name
Patient Middle Initial
Patient Date of Birth
Patient Gender
Patient Driver's Licence Number
Patient Unit Number
Patient Street Number
Patient Street Name
Patient PO Box
Patient City/Town/Village
Patient Province
Patient Postal Code
Practitioner Last Name
Practitioner First Name
Practitioner Licence Number
Practitioner Telephone Number
Practitioner Unit Number
Practitioner Street Number
Practitioner Street Name
Practitioner City/Town/Village
Practitioner Province
Practitioner Postal Code
Practitioner Role
Patient is aware of this report
Notify if patient requests copy
Practitioner's Signature
Date of Report Examination

Medical Document:
""" + document_text
        
        response = model.generate_content(prompt)
        return response.text
        
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
        "Notify if patient requests copy", "Practitioner's Signature", "Date of Report Examination"
    ]
    result = []
    
    # Split the AI response into lines and clean them
    lines = [line.strip() for line in ai_text.split('\n') if line.strip()]
    
    for field in fields:
        value = "Not specified"
        for line in lines:
            # Check if line starts with the field name (case insensitive)
            if line.lower().startswith(field.lower()):
                # Extract the value after the colon
                if ':' in line:
                    value = line.split(':', 1)[1].strip()
                    # Remove any extra formatting
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
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
                "Notify if patient requests copy", "Practitioner's Signature", "Date of Report Examination"
            ])
        writer.writerow(row)

def save_to_json(extracted_text, ai_analysis, filename="extracted_data.json"):
    """Save extracted text and AI analysis to JSON file"""
    import json
    from datetime import datetime
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "complete_extracted_text": extracted_text,
        "ai_analysis": ai_analysis,
        "parsed_fields": {}
    }
    
    # If we have AI analysis, try to parse it into structured data
    if ai_analysis and not ai_analysis.startswith("Gemini API"):
        fields = [
            "Patient Last Name", "Patient First Name", "Patient Middle Initial", "Patient Date of Birth",
            "Patient Gender", "Patient Driver's Licence Number", "Patient Unit Number", "Patient Street Number",
            "Patient Street Name", "Patient PO Box", "Patient City/Town/Village", "Patient Province",
            "Patient Postal Code", "Practitioner Last Name", "Practitioner First Name", "Practitioner Licence Number",
            "Practitioner Telephone Number", "Practitioner Unit Number", "Practitioner Street Number",
            "Practitioner Street Name", "Practitioner City/Town/Village", "Practitioner Province",
            "Practitioner Postal Code", "Practitioner Role", "Patient is aware of this report",
            "Notify if patient requests copy", "Practitioner's Signature", "Date of Report Examination"
        ]
        
        parsed_row = parse_ai_response(ai_analysis)
        for i, field in enumerate(fields):
            data["parsed_fields"][field] = parsed_row[i] if i < len(parsed_row) else "Not specified"
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Complete extracted text and analysis saved to {filename}")
    return filename

def save_raw_text_to_json(extracted_text, filename="raw_extracted_text.json"):
    """Save only the raw extracted text to a separate JSON file"""
    import json
    from datetime import datetime
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "raw_extracted_text": extracted_text,
        "text_length": len(extracted_text),
        "extraction_method": "PyPDF2 + Gemini Vision (if needed)"
    }
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Raw extracted text saved to {filename}")
    return filename

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
        
        # Always save raw extracted text to JSON
        print("Saving raw extracted text to JSON...")
        raw_json_filename = save_raw_text_to_json(text)
        print(f"Raw text saved to {raw_json_filename}")
        
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
                
                # Also save to JSON
                print("Saving to JSON...")
                json_filename = save_to_json(text, result)
                print(f"JSON saved to {json_filename}")
                
                return jsonify({'success': True, 'result': result, 'json_file': json_filename})
            else:
                # Gemini analysis failed, return extracted text only
                print(f"Gemini analysis failed: {result}")
                # Save extracted text to JSON even if analysis failed
                print("Saving extracted text to JSON...")
                json_filename = save_to_json(text, "Analysis failed")
                print(f"JSON saved to {json_filename}")
                
                return jsonify({
                    'success': True, 
                    'result': f"Text extracted successfully but Gemini analysis failed.\n\nExtracted Text:\n{text[:1000]}...",
                    'warning': 'Gemini analysis failed. Only text extraction completed.',
                    'json_file': json_filename
                })
        else:
            # No Gemini API available, return extracted text only
            print("No Gemini API available for analysis")
            # Save extracted text to JSON
            print("Saving extracted text to JSON...")
            json_filename = save_to_json(text, "No API available")
            print(f"JSON saved to {json_filename}")
            
            return jsonify({
                'success': True, 
                'result': f"Text extracted successfully using PyPDF2.\n\nExtracted Text:\n{text[:1000]}...",
                'warning': 'No Gemini API available. Only text extraction completed.',
                'json_file': json_filename
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