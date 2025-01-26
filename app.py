from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2
import numpy as np
import os
import requests
from datetime import timedelta
from datetime import datetime
import base64
from io import BytesIO
import hashlib

import pytesseract
from openai import OpenAI
from dotenv import load_dotenv

# Constants
AUDIO_DIR = "AUDIO_DIR"

def cleanup_audio_cache(max_age_days=7, max_files=100):
    """Remove audio files older than max_age_days and keep only max_files most recent files"""
    try:
        files = []
        for file in os.listdir(AUDIO_DIR):
            if file.endswith('.mp3'):
                path = os.path.join(AUDIO_DIR, file)
                files.append((path, os.path.getmtime(path)))
        
        # Sort files by modification time (newest first)
        files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old files
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for file_path, mtime in files:
            # Remove if too old
            if current_time - mtime > max_age_seconds:
                os.remove(file_path)
                continue
                
            # Remove if we have too many files
            if len(files) > max_files:
                os.remove(file_path)
                
    except Exception as e:
        print(f"Error during cache cleanup: {e}")

def init_app(app):
    """Initialize the application"""
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    cleanup_audio_cache()

# Load environment variables at the start of the app
load_dotenv()

# Create Flask app and initialize
app = Flask(__name__)
init_app(app)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

elevenlabs_api_key = os.getenv("ELEVENLABS_KEY")

@app.route('/')
def index():
	return render_template('index.html')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/ocr', methods=['POST'])
def perform_ocr():
	# Check if file is present in the request
	if 'file' not in request.files:
		return jsonify({'error': 'No file part'}), 400
	
	file = request.files['file']
	
	# Check if filename is empty
	if file.filename == '':
		return jsonify({'error': 'No selected file'}), 400
	
	if file and allowed_file(file.filename):
		# Read the image file
		image = Image.open(file.stream)
		
		# Convert PIL Image to OpenCV format
		cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
		
		# Get image dimensions
		height, width = cv_image.shape[:2]
		
		# Check if coordinates for the marked area are provided
		x1 = request.form.get('x1', type=int)
		y1 = request.form.get('y1', type=int)
		x2 = request.form.get('x2', type=int)
		y2 = request.form.get('y2', type=int)
		
		# If coordinates are provided and valid, crop the image
		if all(coord is not None for coord in [x1, y1, x2, y2]):
			x1, x2 = sorted([x1, x2])
			y1, y2 = sorted([y1, y2])
			if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
				cv_image = cv_image[y1:y2, x1:x2]
			else:
				return jsonify({'error': 'Invalid coordinates'}), 400
			
		# Preprocess the image
		gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		denoised = cv2.fastNlMeansDenoising(gray)
		threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		
		# Perform OCR
		text = pytesseract.image_to_string(threshold)
		
		print ("OCR")
		print (text)
		
		#if not trimmed text is "" then do not fix it
		if text.strip() == "":
			return jsonify({'error': 'No text found'}), 400
		
		prompt = f"""Fix this text but keep the grammar and wording and reply with the fixed text only: \n\n {text}"""
		
		print (prompt)
		model = "gpt-4o"
		messages = [{"role": "user", "content": prompt}]
		
		chat_completion = client.chat.completions.create(
			model=model, 
			messages=messages
		)
		text = chat_completion.choices[0].message.content
		
		print(text)
		return jsonify({
			'text': text.strip(),
			'width': width,
			'height': height
		}), 200
	else:
		return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/translate', methods=['POST'])
def translate_text():
	try:
		data = request.get_json()
		if not data or 'text' not in data or 'language' not in data:
			return jsonify({'error': 'Missing text or language'}), 400
		
		text = data['text']
		language = data['language']
		
		if not text or not language:
			return jsonify({'error': 'Text and language cannot be empty'}), 400
			
		print(f"Translating to {language}: {text}")
		
		prompt = f"""translate this text to {language}. Reply with only the translated text: \n\n {text}"""
		
		print(prompt)
		model = "gpt-4"
		messages = [{"role": "user", "content": prompt}]
		
		chat_completion = client.chat.completions.create(
			model=model, 
			messages=messages
		)
		translated_text = chat_completion.choices[0].message.content
		
		if not translated_text:
			return jsonify({'error': 'Translation failed'}), 500
			
		return jsonify({'translatedText': translated_text})
	except Exception as e:
		print(f"Translation error: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/api/simplify', methods=['POST'])
def simplify_text():
	data = request.get_json()
	if not data or 'text' not in data:
		return jsonify({'error': 'Missing text'}), 400
	
	text = data['text']
	
	prompt = f"""simplify this text and make it easier to understand and use the same language as the original text: \n\n {text}"""
	
	print (prompt)
	model = "gpt-4o"
	messages = [{"role": "user", "content": prompt}]
	
	chat_completion = client.chat.completions.create(
		model=model, 
		messages=messages
	)
	simplified_text = chat_completion.choices[0].message.content
	
	return jsonify({'simplifiedText': simplified_text})

def get_audio_filename(text, voice_id):
    """Generate a filename based on MD5 hash of the text and voice_id combination"""
    # Create MD5 hash of the text + voice_id
    hash_input = f"{text}_{voice_id}"
    text_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    return os.path.join(AUDIO_DIR, f"{text_hash}.mp3")

def text_to_speech(text, voice_id):
    # Check if we already have this audio cached
    audio_file = get_audio_filename(text, voice_id)  # Pass voice_id here
    
    if os.path.exists(audio_file):
        print(f"Using cached audio file: {audio_file}")
        return audio_file
        
    # If not cached, generate new audio
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "language_code": "sv",
        "voice_settings": {
            "stability": 0.31,
            "similarity_boost": 0.97,
            "style": 0.50,
            "use_speaker_boost": True
        },
        "seed": 123,
    }
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": f"{elevenlabs_api_key}"
    }
    
    print(f"Generating new audio for: {text[:50]}...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        # Save the audio file with the hash-based filename
        with open(audio_file, 'wb') as f:
            f.write(response.content)
        print(f"Created new audio file: {audio_file}")
        return audio_file
    else:
        print(f"Error generating audio: {response.status_code}, {response.text}")
        return None

@app.route('/api/text-to-speech', methods=['POST'])
def generate_speech():
	data = request.get_json()
	if not data or 'text' not in data:
		return jsonify({'error': 'No text provided'}), 400
	
	text = data['text']
	print (text)
	voice_id = data.get('voice_id', 'YREPt7KOziuJoYyc1RTB')  # Default voice ID
	
	#try:
	audio_file = text_to_speech(text, voice_id)
	print (audio_file)
	if audio_file:
		return send_file(audio_file, mimetype='audio/mpeg')
	else:
		return jsonify({'error': 'Failed to generate speech'}), 500
	#except Exception as e:
	#	return jsonify({'error': str(e)}), 500

@app.route('/api/explain', methods=['POST'])
def explain_image():
	data = request.get_json()
	if not data or 'image' not in data:
		return jsonify({'error': 'No image provided'}), 400
	
	# Get base64 image from request
	image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
	
	# Get vision analysis
	response = client.chat.completions.create(
		model="gpt-4o",
		messages=[
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "Explain how to solve this. if there is text in the image, explain using the same language as the text."
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{image_data}"
						}
					}
				]
			}
		],
		max_tokens=300
	)
	
	explanation = response.choices[0].message.content
	return jsonify({'explanation': explanation})

# Add this new route to fetch available voices
@app.route('/api/voices', methods=['GET'])
def get_voices():
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "Accept": "application/json",
        "xi-api-key": elevenlabs_api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            voices = response.json()['voices']
            # Return only necessary info
            voice_list = [{'id': v['voice_id'], 'name': v['name'], 'preview_url': v.get('preview_url')} 
                         for v in voices]
            return jsonify(voice_list)
        else:
            return jsonify({'error': 'Failed to fetch voices'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
	