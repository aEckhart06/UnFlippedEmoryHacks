from flask import Flask, render_template, request, url_for, redirect, Response, jsonify
import pymongo
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import gridfs
from dotenv import load_dotenv
from bson import ObjectId
import cv2
import numpy as np
import io
import tempfile
import os
from Chat import call_model
from Transcribe import transcribe
from datetime import datetime
import threading
from GenerateQuestions import generate_mcq, ask_question, split_transcript_by_time
from functools import wraps


load_dotenv()

app = Flask(__name__)

client = MongoClient('localhost', 27017)


# Our MongoDB Database
db = client.flask_database

# Our MongoDB Collection
video_collection = db.videos
transcript_collection = db.transcripts

# Our GridFS Bucket
fs = gridfs.GridFS(db)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Will show data and allow to submit data (with the methods)
@app.route('/')
def index():
    return render_template('index.html')


# Example protected route
@app.route('/library')
def library():
    # Only authenticated users can access this route
    files = fs.find()

    video_list = []
    for file in files:
        # Check if transcript exists for this video
        transcript = transcript_collection.find_one({'file_id': file._id})
        video_list.append({
            'filename': file.filename,
            'upload_date': file.upload_date,
            'file_id': str(file._id),
            'transcript_ready': transcript is not None  # Boolean flag for transcript status
        })
    return render_template('library.html', video_list=video_list)

@app.route('/videoPage/<file_id>')
def videoPage(file_id):
    video_url = url_for('get_video', file_id=file_id)
    transcript = transcript_collection.find_one({'file_id': ObjectId(file_id)})['transcript']
    return render_template('videoPage.html', video_url=video_url, transcript=transcript)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    filename = secure_filename(file.filename)
    # Save file to GridFS
    file_id = fs.put(file, filename=filename)
    
    # Start transcription in a separate thread
    thread = threading.Thread(target=process_transcript, args=(file_id, filename))
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()
    
    return redirect(url_for('library'))

def process_transcript(file_id: ObjectId, filename: str):
    try:
        grid_out = fs.get(file_id)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(grid_out.read())
            temp_path = temp_file.name
        
        try:
            # Pass the file path instead of the GridOut object
            transcript = transcribe(temp_path)
            
            transcript_doc = {
                'file_id': file_id,
                'filename': filename,
                'transcript': transcript,
                'created_at': datetime.now()
            }
            transcript_collection.insert_one(transcript_doc)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")

@app.route('/video/<file_id>')
def get_video(file_id):
    file = fs.get(ObjectId(file_id))
    return Response(file, content_type='video/mp4')

@app.route('/delete/<file_id>', methods=['POST'])
def delete_video(file_id):
    try:
        # Convert string ID to ObjectId and delete the file from GridFS
        fs.delete(ObjectId(file_id))
        return redirect(url_for('library'))
    except Exception as e:
        print(f"Error deleting file: {str(e)}")  # Add logging
        return f'Error deleting file: {str(e)}', 500
    
@app.route('/submitPrompt', methods=['POST'])
def submitPrompt():
    transcript = request.json['transcript']
    current_timestamp = request.json['current_timestamp']
    prompt = request.json['prompt']
    response = call_model(transcript, current_timestamp, prompt)
    print(response)
    
    return jsonify({'response': response})
    


@app.route('/get_first_frame/<file_id>')
def get_first_frame(file_id):
    try:
        print(f"Attempting to get first frame for file_id: {file_id}")
        file = fs.get(ObjectId(file_id))
        video_data = file.read()
        print(f"Successfully read video data, size: {len(video_data)} bytes")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_data)
            temp_path = temp_file.name
            print(f"Created temporary file at: {temp_path}")
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(temp_path)
            print("Created VideoCapture object")
            
            if not cap.isOpened():
                print(f"Failed to open video capture for file: {temp_path}")
                # Try alternative method using numpy array
                try:
                    print("Attempting alternative method using numpy array")
                    video_bytes = np.frombuffer(video_data, np.uint8)
                    cap = cv2.VideoCapture()
                    cap.open(io.BytesIO(video_bytes))
                    
                    if not cap.isOpened():
                        print("Both methods failed to open video")
                        return "Failed to open video", 500
                except Exception as e:
                    print(f"Alternative method failed: {str(e)}")
                    return "Failed to open video", 500
                
            # Read the first frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                return "Failed to read frame", 500
                
            print(f"Successfully read frame, shape: {frame.shape}")
            
            # Convert frame to JPEG with specific quality
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_data = buffer.tobytes()
            print(f"Converted frame to JPEG, size: {len(frame_data)} bytes")
            
            # Release the video capture
            cap.release()
            print("Released video capture")
            
            # Set cache control headers
            response = Response(frame_data, content_type='image/jpeg')
            response.headers['Cache-Control'] = 'public, max-age=31536000'
            return response
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
                print("Cleaned up temporary file")
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")
                
    except Exception as e:
        print(f"Error extracting first frame: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}", 500

@app.route('/check_processing_status')
def check_processing_status():
    # Get all videos that have completed processing since page load
    files = fs.find()
    completed_videos = []
    for file in files:
        transcript = transcript_collection.find_one({'file_id': file._id})
        if transcript:
            completed_videos.append(str(file._id))
    
    return jsonify({'completed_videos': completed_videos})

@app.route('/createQuestion', methods=['POST'])
def createQuestion():
    transcript = request.json['transcript']
    current_timestamp = request.json['current_timestamp']
    current_timestamp = float(current_timestamp)
    interval = request.json['interval']
    interval = float(interval)
    interval_seconds = interval * 60
    segments = split_transcript_by_time(transcript, interval_seconds)
    if len(segments) > 0:
        i = int(current_timestamp / (interval_seconds))
        question_object = generate_mcq(segments[i])
        print(question_object)
    else:
        question_object = None

    return jsonify({'question_object': question_object})

@app.route('/rename/<file_id>', methods=['POST'])
def rename_file(file_id):
    try:
        # Get the new filename from the form submission
        new_filename = request.form.get('new_filename')
        
        if not new_filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Secure the filename to prevent any security issues
        new_filename = secure_filename(new_filename)
        
        # Add extension if it's missing (assuming mp4 as default)
        if not new_filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
            new_filename += '.mp4'
        
        # Update filename in GridFS
        # Note: GridFS doesn't allow direct updates, so we need to get the file
        # and re-save it with the new filename
        file_obj = fs.get(ObjectId(file_id))
        old_file_data = file_obj.read()
        
        # Store old metadata
        old_metadata = {}
        for key in file_obj._file:
            if key not in ['_id', 'filename', 'uploadDate']:
                old_metadata[key] = file_obj._file[key]
        
        # Delete old file
        fs.delete(ObjectId(file_id))
        
        # Create new file with same ID if possible, or get the new ID
        new_file_id = fs.put(old_file_data, filename=new_filename, **old_metadata)
        
        # Update transcript collection if there's a transcript for this file
        transcript_collection.update_one(
            {'file_id': ObjectId(file_id)},
            {'$set': {'file_id': new_file_id, 'filename': new_filename}}
        )
        
        return redirect(url_for('library'))
        
    except Exception as e:
        print(f"Error renaming file: {str(e)}")
        return f'Error renaming file: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)
