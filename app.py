import cv2
import base64
import time
import threading
import os
import wave
import pyaudio
from flask import Flask, render_template, Response, jsonify, request
from groq import Groq
from rtwhisper import transcribe_audio 

app = Flask(__name__)

camera = None           
processing_active = False
processing_thread = None
unique_results = []    
camera_lock = threading.Lock()  

audio_recording_active = False
audio_thread = None
AUDIO_FILENAME = os.path.join(os.path.dirname(__file__), "audio.m4a")

client = Groq(api_key="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")
think = Groq(api_key="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")
bro = Groq(api_key="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")

_ = bro.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You have been given eyes and ears to listen. Have a conversation or generate thoughts based on what the user shows you and says to you."
        }
    ],
    temperature=1,
    top_p=1,
    stream=True,
    stop=None,
)


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

def process_frames():
    global camera, unique_results, processing_active
    while processing_active:
        with camera_lock:
            current_camera = camera
        
        if current_camera is None:
            time.sleep(0.1)
            continue
        
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

def gen_frames():
    while True:
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

# ---------------------
# Audio Recording Section
# ---------------------

def record_audio_thread():
    """
    Records audio continuously using PyAudio until the global flag is turned off.
    The recorded audio is saved to AUDIO_FILENAME.
    """
    global audio_recording_active
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    print("Audio recording started...")
    while audio_recording_active:
        try:
            data = stream.read(CHUNK)
            frames.append(data)
        except Exception as e:
            print("Audio recording error:", e)
            break
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Save the recorded data to a file.
    # (Note: Using wave to write a file with an .m4a extension; adjust as needed for your setup.)
    wf = wave.open(AUDIO_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Audio recording saved to", AUDIO_FILENAME)

# ---------------------
# Flask Routes
# ---------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, processing_active, processing_thread, audio_recording_active, audio_thread
    with camera_lock:
        if camera is None:
            camera = VideoCamera()
            processing_active = True
            processing_thread = threading.Thread(target=process_frames, daemon=True)
            processing_thread.start()
    # Start audio recording as well
    if not audio_recording_active:
        audio_recording_active = True
        audio_thread = threading.Thread(target=record_audio_thread, daemon=True)
        audio_thread.start()
    return jsonify({'status': 'camera and audio recording started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, processing_active, processing_thread, audio_recording_active, audio_thread
    processing_active = False
    if processing_thread is not None:
        processing_thread.join()
        processing_thread = None
    with camera_lock:
        if camera is not None:
            camera.stop()
            camera = None
    # Stop audio recording if it is still active
    if audio_recording_active:
        audio_recording_active = False
        if audio_thread is not None:
            audio_thread.join()
            audio_thread = None
    return jsonify({'status': 'camera and audio recording stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/summary')
def summary():
    global unique_results, audio_recording_active, audio_thread
    # Ensure audio recording is stopped so the file is complete.
    if audio_recording_active:
        audio_recording_active = False
        if audio_thread is not None:
            audio_thread.join()
            audio_thread = None

    # Transcribe the audio from the saved file.
    try:
        transcription_result = transcribe_audio(AUDIO_FILENAME)
    except Exception as e:
        transcription_result = f"Audio transcription failed: {e}"

    # Combine the unique video observations.
    combined_responses = "\n".join(unique_results)

    # Create an enhanced prompt for the bro model.
    enhanced_prompt = (
        "You are a perceptive AI with both visual and auditory insights. "
        "Based on the following observations, provide a creative, thoughtful, and concise and casual commentary , you can crack jokes . "
        "Highlight interesting details, potential interpretations, and any hidden context you infer. "
        "\n\nVideo Observations:\n" + combined_responses +
        "\n\nAudio Transcription:\n" + transcription_result
    )
    messages = [
        {
            "role": "user",
            "content": enhanced_prompt
        }
    ]
    try:
        bro_response = bro.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        response_text = bro_response.choices[0].message.content
        return jsonify({'summary': response_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5034, debug=True, use_reloader=False)