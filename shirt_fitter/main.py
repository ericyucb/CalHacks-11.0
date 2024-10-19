import cv2
import mediapipe as mp
import numpy as np
from meshresizer import resize, overlay_image

global clothesimage
clothesimage = cv2.imread("clothes2transparent.png", cv2.IMREAD_UNCHANGED)
clothesimage = cv2.resize(clothesimage, (440, 667))
def convert(p):  # point in 0-1 form
    return [p[0] * WIDTH, p[1] * HEIGHT]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cam = cv2.VideoCapture(0)

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True)
while True:
    WIDTH = 1280
    HEIGHT = 720
    ret, frame = cam.read()
    # Recolor image to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (WIDTH, HEIGHT))

    # Make detection
    results_pose = pose.process(img)
    res_pose = results_pose.pose_landmarks
    if res_pose:
        pose_landmark = results_pose.pose_landmarks.landmark
        # these positions are between 0-1
        shoulderL = np.array([pose_landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     pose_landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        shoulderR = np.array([pose_landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     pose_landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        hipL = np.array([pose_landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                pose_landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        hipR = np.array([pose_landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                pose_landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

        hipToShoulderL = shoulderL - hipL
        hipToShoulderR = shoulderR - hipR
        shirtTopScale = 0.1
        topL = convert(shoulderL + shirtTopScale * hipToShoulderL)
        topR = convert(shoulderR + shirtTopScale * hipToShoulderR)
        img = cv2.circle(img, (int(topL[0]), int(topL[1])), radius=5, color=(0, 255, 0), thickness=-1)
        img = cv2.circle(img, (int(topR[0]), int(topR[1])), radius=5, color=(0, 255, 0), thickness=-1)

        hipBottomScale = 0.4
        hipToHip = hipL - hipR
        bottomL = convert(hipL + hipBottomScale * hipToHip)
        bottomR = convert(hipR - hipBottomScale * hipToHip)
        img = cv2.circle(img, (int(bottomL[0]), int(bottomL[1])), radius=5, color=(255, 0, 0), thickness=-1)
        img = cv2.circle(img, (int(bottomR[0]), int(bottomR[1])), radius=5, color=(255, 0, 0), thickness=-1)

        # (0,0) is top left corner for pixel position. x increases going right and y increases going down
        # bottomL is a list [x,y] of pixel positions for left hip
        # bottomR is a list [x,y] of pixel positions for right hip
        # topL is a list [x,y] of pixel positions for left shoulder
        # topR is a list [x,y] of pixel positions for right shoulder
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # warped_image_rgba, pivot = resize((topL[0],topL[1]-clothesimage.shape[0]), (topR[0],topR[1]-clothesimage.shape[0]), (bottomL[0],bottomL[1]-clothesimage.shape[0]), (bottomR[0], bottomR[1]-clothesimage.shape[0]), clothesimage)
        mult = 1.9 if abs(topL[0]-topR[0]>370) else 0.00179174*(abs(topL[0]-topR[0])) + 1.2224
        warped_image_rgba, pivot = resize((topL[0], bottomL[1]), (topR[0], bottomR[1]), (bottomL[0],mult*bottomL[1]), (bottomR[0], mult*bottomR[1]), clothesimage)
        img = overlay_image(img, warped_image_rgba, pivot)
        # Recolor back to BGR
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img.flags.writeable = True

    # Render detections
    mp_drawing.draw_landmarks(img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('Mediapipe Feed', img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

