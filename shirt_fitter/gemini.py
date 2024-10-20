import os
import google.generativeai as genai
import cv2
import json

genai.configure(api_key='AIzaSyBDhYAAOLh8HNWXOsXZRXEomgH_jlZbZt4')


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

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

# loop for video
path = 'shirt_fitter/gemini_images'  # path to folder of images
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (640, 360))
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('0'):
        print('capturing image')
        img_path = path + f'/test0.png'
        cv2.imwrite(img_path, frame)  # add to folder
        print(img_path)

        prompt_parts = [
            upload_to_gemini(img_path, mime_type="image/png"),
            "You are a fashion expert who gives fashion advice to people. Your job is to analyze a photo of me and give me 3 specific recommendations of t-shirts from various brands. Output the 3 clothes as a JSON dictionary that includes the brand, shirt type, and color. Also, provide a concise and simple description of what shirt I am wearing. Reference me in 2nd person."
        ]

        try:
            response = model.generate_content(prompt_parts)
            deez = json.loads(response.text)
            description = deez['description']
            recs_dict = deez['recommendations']
            recs = [f'{n['brand']} {n['color']} {n['shirt_type']}' for n in recs_dict]
        except:
            print('Please try again.')

        print(description)  # a string description of what shirt the person in the image is wearing
        print(recs)  # array of 3 strings. each string is a shirt recommendation


    if cv2.waitKey(1) == ord('q'):
        break
