import cv2
import base64
import time
import threading
from groq import Groq

client = Groq(api_key="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")

unique_results = []

latest_frame = None
frame_lock = threading.Lock()

processing_active = True

def process_frames():
    global latest_frame, unique_results, processing_active
    while processing_active:
        frame_to_process = None


        with frame_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
                latest_frame = None

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
                print(f"API Error during frame processing: {str(e)}")
        
        else:
            time.sleep(0.01)

processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

cap = cv2.VideoCapture(0)
print("Starting video capture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    with frame_lock:
        latest_frame = frame.copy()

    cv2.imshow("Video Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
processing_active = False
cap.release()
cv2.destroyAllWindows()
processing_thread.join(timeout=2)
print("\nUnique responses collected:")
for idx, res in enumerate(unique_results, 1):
    print(f"{idx}. {res}")
combined_responses = "\n".join(unique_results)
summary_message = [{
    "role": "user",
    "content": f"Summarize these observations into bullet points:\n{combined_responses}"
}]

print("\nGenerating summary...")

try:
    summary_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=summary_message,
        temperature=0.6,
        top_p=0.95,
        stream=True,
    )

    print("Summary:")
    summary_text = ""
    for chunk in summary_completion:
        text_chunk = chunk.choices[0].delta.content or ""
        print(text_chunk, end="")
        summary_text += text_chunk

    print("\n\nSummary complete.")

except Exception as e:
    print("Summary Error:", e)