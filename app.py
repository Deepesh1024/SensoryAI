import cv2
import base64
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
from groq import Groq
from rtwhisper import TranscriptionApp
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate

app = Flask(__name__)

# def audio():
#     app = TranscriptionApp(
#         env_variable_name="GROQ_API_KEY",
#         output_filename="temp.wav",
#         pause_threshold=2.0
#     )
#     app.run()

# Global variables
camera = None           # Holds our VideoCamera instance
processing_active = False
processing_thread = None
unique_results = []     # Stores unique vision responses
camera_lock = threading.Lock()  # Lock for camera access

# Initialize the Groq client with your API key
client = Groq(api_key="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")
# think = Groq(api_key="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")

# completion = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {
#             "role": "system",
#             "content": "i will give you some video summarisation and audio transcription you have to think and respond what is happening around you\n"
#         }
#     ],
#     temperature=1,
#     max_completion_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")



# VideoCamera class to capture frames from the webcam
class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.update_thread = threading.Thread(target=self.update, daemon=True)
        self.update_thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                self.running = False
                break

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.frame)
                if ret:
                    return jpeg.tobytes()
        return None

    def stop(self):
        self.running = False
        self.update_thread.join()  # Wait for update thread to finish
        self.cap.release()

# Background frame processing
def process_frames():
    global camera, unique_results, processing_active
    while processing_active:
        # Get camera reference safely
        with camera_lock:
            current_camera = camera
        
        if current_camera is None:
            time.sleep(0.1)
            continue
        
        # Get frame safely
        with current_camera.lock:
            frame_to_process = current_camera.frame.copy() if current_camera.frame is not None else None
        
        if frame_to_process is not None:
            try:
                ret_enc, buffer = cv2.imencode('.jpg', frame_to_process)
                if ret_enc:
                    image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe what you see in one sentence"},
                            {"type": "image_url",
                             "image_url": {
                                 "url": f"data:image/jpeg;base64,{image_base64}",
                                 "detail": "high"
                             }}
                        ]
                    }]
                    completion = client.chat.completions.create(
                        model="llama-3.2-90b-vision-preview",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=300,
                        timeout=10
                    )
                    response_text = completion.choices[0].message.content
                    if response_text not in unique_results:
                        unique_results.append(response_text)
                        print("Vision Response:", response_text)
            except Exception as e:
                print(f"API Error: {str(e)}")
        
        time.sleep(0.5)  # Process ~2 frames/sec

# MJPEG video streaming generator
def gen_frames():
    while True:
        # Safely check camera status
        with camera_lock:
            if camera is None:
                break
            current_camera = camera
        
        frame = current_camera.get_frame()
        if frame is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # ~30 fps

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, processing_active, processing_thread
    with camera_lock:
        if camera is None:
            camera = VideoCamera()
            processing_active = True
            processing_thread = threading.Thread(target=process_frames, daemon=True)
            processing_thread.start()
    return jsonify({'status': 'camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, processing_active, processing_thread
    processing_active = False

    # Wait for processing thread to finish
    if processing_thread is not None:
        processing_thread.join()
        processing_thread = None

    # Clean up camera resources
    with camera_lock:
        if camera is not None:
            camera.stop()
            camera = None

    return jsonify({'status': 'camera stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/summary')
def summary():
    combined_responses = "\n".join(unique_results)
    summary_message = [{
        "role": "user",
        "content": f"Summarize these observations into bullet points:\n{combined_responses}"
    }]
    try:
        summary_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=summary_message,
            temperature=0.6,
            top_p=0.95,
            stream=False
        )
        summary_text = summary_completion.choices[0].message.content
        return jsonify({'summary': summary_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5034, debug=True, use_reloader=False)