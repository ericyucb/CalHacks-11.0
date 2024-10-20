import os
import google.generativeai as genai
import cv2
import json
import time
import math

genai.configure(api_key='AIzaSyBDhYAAOLh8HNWXOsXZRXEomgH_jlZbZt4')

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def gemini_run(model, frame):
    img_path = 'shirt_fitter/gemini_images/test0.png'
    cv2.imwrite(img_path, frame)

    prompt_parts = [
        upload_to_gemini(img_path, mime_type="image/png"),
        "You are a fashion expert who gives fashion advice to people. Your job is to analyze a photo of me and give me 3 specific recommendations of t-shirts from various brands. Output the 3 clothes as a JSON dictionary that includes the brand, shirt type, and color. Please suggest darker colored clothing. Also, provide a concise and simple description of what shirt I am wearing. Reference me in 2nd person."
    ]

    try:
        response = model.generate_content(prompt_parts)
        deez = json.loads(response.text)
        description = deez['description']
        recs_dict = deez['recommendations']
        recs = [f'{n["brand"]} {n["color"]} {n["shirt_type"]}' for n in recs_dict]
        return description, recs
    except:
        return None

def aura():
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
    )

    # Video capture loop
    path = 'shirt_fitter/gemini_images'
    cam = cv2.VideoCapture(0)
    timer = 6.5
    prev_time = time.perf_counter()
    
    while True:
        current_time = time.perf_counter()
        dt = current_time - prev_time
        timer -= dt

        ret, frame = cam.read()
        frame = cv2.resize(frame, (640, 360))

        # Perform action after timer
        if timer < 0:
            output = gemini_run(model, frame)
            if output:
                return output
                description, recs = output
                print(description)
                print(recs)
                break
            else:
                return False
                print('Try again')

        # Display the countdown or "Capturing" message
   
        if math.floor(timer) == 0:
            text = 'Capturing!'
            text_position = (50, 50)
            font_size = 1.2
        else:
            frame = cv2.putText(frame, 'Get Ready...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            text_position = (280, 200)
            text = str(math.floor(timer))
            font_size = 3
            
        frame = cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                            (255, 255, 255), 3, cv2.LINE_AA) 

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        prev_time = current_time

    cv2.destroyAllWindows()