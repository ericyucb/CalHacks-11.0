import torch
from torchvision import models
import cv2
import numpy as np
import torchvision.transforms as T

# Load the pre-trained DeepLabV3 model for semantic segmentation
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Define the transformation to prepare the input image
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def decode_segmap(image, source_image, nc=21):
    # Define a color for each class (21 classes in the PASCAL VOC dataset)
    label_colors = np.array([(0, 0, 0),         # 0=background
                             (128, 0, 0),       # 1=aeroplane
                             (0, 128, 0),       # 2=bicycle
                             (128, 128, 0),     # 3=bird
                             (0, 0, 128),       # 4=boat
                             (128, 0, 128),     # 5=bottle
                             (0, 128, 128),     # 6=bus
                             (128, 128, 128),   # 7=car
                             (64, 0, 0),        # 8=cat
                             (192, 0, 0),       # 9=chair
                             (64, 128, 0),      # 10=cow
                             (192, 128, 0),     # 11=dining table
                             (64, 0, 128),      # 12=dog
                             (192, 0, 128),     # 13=horse
                             (64, 128, 128),    # 14=motorbike
                             (192, 128, 128),   # 15=person (this is the torso class we are interested in)
                             (0, 64, 0),        # 16=potted plant
                             (128, 64, 0),      # 17=sheep
                             (0, 192, 0),       # 18=sofa
                             (128, 192, 0),     # 19=train
                             (0, 64, 128)])     # 20=tv/monitor

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        if l < len(label_colors):  # Avoid index out of bounds error
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return cv2.addWeighted(source_image, 0.6, rgb, 0.4, 0)



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the DeepLabV3 model
    input_image = preprocess(frame).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Decode the output and overlay it on the original frame
    segmented_image = decode_segmap(output_predictions, frame)

    # Display the original frame and the segmented image
    cv2.imshow('Segmentation', segmented_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

