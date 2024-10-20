import cv2
import numpy as np
import time
import mediapipe as mp

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Webcam capture
cap = cv2.VideoCapture(0)

# Capture frames over a duration
capture_duration = 3  # seconds
start_time = time.time()
frames = []
pose_points = []

def convert(p):  # point in 0-1 form
    return [p[0] * WIDTH, p[1] * HEIGHT]

while int(time.time() - start_time) < capture_duration:
    WIDTH = 1280
    HEIGHT = 720
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT))
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        pose_landmark = result.pose_landmarks.landmark
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
        
        hipBottomScale = 0.4
        hipToHip = hipL - hipR
        bottomL = convert(hipL + hipBottomScale * hipToHip)
        bottomR = convert(hipR - hipBottomScale * hipToHip)
        
        # Get the four points: left shoulder, right shoulder, left hip, right hip
        points = [
            topL,
            topR,
            bottomL,
            bottomR,
        ]
        
        # Save frame and corresponding pose points
        frames.append(frame)
        pose_points.append(points)

# Release the webcam
cap.release()
cv2.destroyAllWindows()

# Convert pose points to numpy array for processing
pose_points = np.array(pose_points)

# Find the average of the points
average_points = np.mean(pose_points, axis=0)

# Calculate the frame whose points are closest to the average
distances = np.linalg.norm(pose_points - average_points, axis=(1, 2))
closest_frame_idx = np.argmin(distances)
closest_frame = frames[closest_frame_idx]
closest_pose_points = pose_points[closest_frame_idx]

# Define the source and destination points for the warp
src_points = np.array([
    [0, 0],  # Top-left corner of the overlay image
    [881, 0],  # Top-right corner of the overlay image
    [0, 1335],  # Bottom-left corner of the overlay image
    [881, 1335]  # Bottom-right corner of the overlay image
], dtype=np.float32)

dst_points = np.array([
    [closest_pose_points[0][0] * closest_frame.shape[1], closest_pose_points[0][1] * closest_frame.shape[0]],
    [closest_pose_points[1][0] * closest_frame.shape[1], closest_pose_points[1][1] * closest_frame.shape[0]],
    [closest_pose_points[2][0] * closest_frame.shape[1], closest_pose_points[2][1] * closest_frame.shape[0]],
    [closest_pose_points[3][0] * closest_frame.shape[1], closest_pose_points[3][1] * closest_frame.shape[0]]
], dtype=np.float32)

# Load the overlay image with transparency (PNG)
overlay_image = cv2.imread('shirt_fitter/clothes2transparent.png', cv2.IMREAD_UNCHANGED)

# Separate the image and the alpha channel
overlay_img = overlay_image[..., :3]  # RGB image
overlay_alpha = overlay_image[..., 3] / 255.0  # Alpha channel scaled between 0 and 1

# Resize the overlay to fit the transformation points
warped_overlay_img = cv2.warpPerspective(overlay_img, cv2.getPerspectiveTransform(src_points, dst_points), 
                                         (closest_frame.shape[1], closest_frame.shape[0]))
warped_overlay_alpha = cv2.warpPerspective(overlay_alpha, cv2.getPerspectiveTransform(src_points, dst_points), 
                                           (closest_frame.shape[1], closest_frame.shape[0]))

# Prepare the alpha mask for blending
warped_overlay_alpha = np.expand_dims(warped_overlay_alpha, axis=-1)

# Blend the overlay with the frame
blended_frame = (warped_overlay_img * warped_overlay_alpha + closest_frame * (1 - warped_overlay_alpha)).astype(np.uint8)

# Display the result
cv2.imshow("Overlay Result", blended_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
