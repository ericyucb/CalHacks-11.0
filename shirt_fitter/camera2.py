import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utils from mediapipe to draw landmarks on frame
mp_drawing = mp.solutions.drawing_utils

# Webcam capture
cap = cv2.VideoCapture(0)

def convert(p):  # point in 0-1 form
    return [p[0] * WIDTH, p[1] * HEIGHT]
def lerp(a,b,t):
    return (1-t)*a + t*b
def llerp(a,b,t):
    return [(1-t)*a[0]+t*b[0],(1-t)*a[1]+t*b[1]]
def angb2vec(v1, v2):
    return np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )

# Load the overlay image with transparency (PNG)
overlay_image = cv2.imread('shirt_fitter/clothe2.PNG', cv2.IMREAD_UNCHANGED)
right_sleeve = cv2.imread('shirt_fitter/rsle.PNG', cv2.IMREAD_UNCHANGED)
left_sleeve = cv2.imread('shirt_fitter/lsle.PNG', cv2.IMREAD_UNCHANGED)
overlay_image = cv2.flip(overlay_image, 0)
# right_sleeve = cv2.flip()
left_sleeve = cv2.flip(left_sleeve, 1)
overlay_img = overlay_image[..., :3]  # RGB image
right_slv = right_sleeve[..., :3]
left_slv = left_sleeve[..., :3]
overlay_alpha = overlay_image[..., 3] / 255.0  # Alpha channel scaled between 0 and 1
right_alpha = right_sleeve[..., 3] / 255.0
left_alpha = left_sleeve[..., 3] / 255.0

while cap.isOpened():
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
        shirtTopScale = 0.2
        
        topL = convert(shoulderL + shirtTopScale * hipToShoulderL)
        topR = convert(shoulderR + shirtTopScale * hipToShoulderR)
        
        hipBottomScale = 0.4
        hipToHip = hipL - hipR
        bottomL = convert(hipL + hipBottomScale * hipToHip)
        bottomR = convert(hipR - hipBottomScale * hipToHip)
        # Get the four points: left shoulder, right shoulder, left hip, right hip
        points = [
            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
        ]
        # theta is angle between hip - shoulder and elbow - shoulder
        hipR = np.array(convert([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]))
        shouldR = np.array(convert([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]))
        elbowR = np.array(convert([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]))
        hipL = np.array(convert([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]))
        shouldL = np.array(convert([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]))
        elbowL = np.array(convert([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]))
        # theta = np.arccos( np.dot(hipR-shouldR, elbowR-shouldR) / (np.linalg.norm(hipR-shouldR) * np.linalg.norm(elbowR-shouldR)) )
        print(hipR, shouldR)
        theta = angb2vec(hipR-shouldR, elbowR-shouldR)
        theta2 = 1.5*np.pi-theta
        ltheta = angb2vec(hipL-shouldL, elbowL-shouldL)
        ltheta2 = ltheta-0.5*np.pi
        # pointstoScaleTo = [topL, topR, bottomL, bottomR]
        pointstoScaleTo = [bottomL, bottomR, topL, topR]
        pointsRightSleeve = [[topR[0], topR[1]+0.1*np.linalg.norm(hipR-shouldR)],
                            #[topR[0] - 0.2*np.linalg.norm(hipR-shouldR), min(topR[1]+0.1*np.linalg.norm(hipR-shouldR), max(llerp(topR, bottomR, 0.45)[1], topR[1]+0.1*np.linalg.norm(hipR-shouldR) - llerp(topR, bottomR, 0.45)[1] + llerp(topR, bottomR, 0.15)[1]-0.25*np.linalg.norm(hipR-shouldR)*np.sin(theta2)))], 
                            [topR[0] - 0.2*np.linalg.norm(hipR-shouldR), 
                                min(topR[1]+0.1*np.linalg.norm(hipR-shouldR), llerp(topR, bottomR, 0.15)[1] - 0.25*np.linalg.norm(hipR-shouldR)*np.sin(theta2) - 0.15*np.linalg.norm(hipR-shouldR))],
                            llerp(topR, bottomR, 0.45), 
                            [llerp(topR, bottomR, 0.15)[0] + 0.25*np.linalg.norm(hipR-shouldR)*np.cos(theta2), llerp(topR, bottomR, 0.15)[1] - 0.25*np.linalg.norm(hipR-shouldR)*np.sin(theta2)]]
        pointsLeftSleeve = [[topL[0], topL[1]+0.1*np.linalg.norm(hipL-shouldL)],
                            [topL[0] + 0.2*np.linalg.norm(hipL-shouldL), 
                                min(topL[1] + 0.1*np.linalg.norm(hipL-shouldL), llerp(topL, bottomL, 0.15)[1] - 0.25*np.linalg.norm(hipL-shouldL)*np.sin(ltheta2) - 0.15*np.linalg.norm(hipL-shouldL))],
                            llerp(topL, bottomL, 0.45),
                            [llerp(topL, bottomL, 0.15)[0] + 0.25*np.linalg.norm(hipL-shouldL)*np.cos(ltheta2), llerp(topL, bottomL, 0.15)[1] - 0.25*np.linalg.norm(hipL-shouldL)*np.sin(ltheta2)]]
        #print(pointsRightSleeve)
        # Scale the points to the frame dimensions
        points = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for (x, y) in points]
        
        # Draw the pose landmarks on the frame
        
        #mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw the points on the frame
        # for point in points:
        #     cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Green circle on points
        
        # Define the source and destination points for the warp
        src_points = np.array([
            [0, 0],  # Top-left corner of the overlay image
            [overlay_img.shape[1], 0],  # Top-right corner of the overlay image
            [0, overlay_img.shape[0]],  # Bottom-left corner of the overlay image
            [overlay_img.shape[1], overlay_img.shape[0]]  # Bottom-right corner of the overlay image
        ], dtype=np.float32)
        src_rsle = np.array([
            [0, 0],
            [right_slv.shape[1], 0],
            [0, right_slv.shape[0]],
            [right_slv.shape[1], right_slv.shape[0]]
        ], dtype=np.float32)
        src_lsle = np.array([
            [0, 0],
            [left_slv.shape[1], 0],
            [0, left_slv.shape[0]],
            [right_slv.shape[1], right_slv.shape[0]]
        ], dtype=np.float32)

        dst_points = np.array(pointstoScaleTo, dtype=np.float32)
        dst_rsle = np.array(pointsRightSleeve, dtype=np.float32)
        dst_lsle = np.array(pointsLeftSleeve, dtype=np.float32)

        # Perform perspective warp
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        N = cv2.getPerspectiveTransform(src_rsle, dst_rsle)
        O = cv2.getPerspectiveTransform(src_lsle, dst_lsle)
        warped_overlay_img = cv2.warpPerspective(overlay_img, M, (frame.shape[1], frame.shape[0]))
        warped_rsle = cv2.warpPerspective(right_slv, N, (frame.shape[1], frame.shape[0]))
        warped_lsle = cv2.warpPerspective(left_slv, O, (frame.shape[1], frame.shape[0]))
        warped_overlay_alpha = cv2.warpPerspective(overlay_alpha, M, (frame.shape[1], frame.shape[0]))
        warped_rsle_alpha = cv2.warpPerspective(right_alpha, N, (frame.shape[1], frame.shape[0]))
        warped_lsle_alpha = cv2.warpPerspective(left_alpha, O, (frame.shape[1], frame.shape[0]))

        # Prepare the alpha mask for blending
        warped_overlay_alpha = np.expand_dims(warped_overlay_alpha, axis=-1)
        warped_rsle_alpha = np.expand_dims(warped_rsle_alpha, axis=-1)
        warped_lsle_alpha = np.expand_dims(warped_lsle_alpha, axis=-1)

        # Blend the overlay with the frame
        frame = (warped_overlay_img * warped_overlay_alpha + frame * (1 - warped_overlay_alpha)).astype(np.uint8)
        frame = (warped_rsle * warped_rsle_alpha + frame * (1 - warped_rsle_alpha)).astype(np.uint8)
        frame = (warped_lsle * warped_lsle_alpha + frame * (1 - warped_lsle_alpha)).astype(np.uint8)

        # for point in pointstoScaleTo:
        #     cv2.circle(frame, (round(point[0]), round(point[1])), 5, (255, 255, 255), -1)  # Green circle on points
        # for i, point in enumerate(pointsLeftSleeve):
        #     cv2.circle(frame, (round(point[0]), round(point[1])), 10, (255,0,0), -1)
        # for i, point in enumerate(pointsRightSleeve):
        #     cv2.circle(frame, (round(point[0]), round(point[1])), 10, (255,0,0), -1)
        #cv2.circle(frame, (round(pointsRightSleeve[3][0]), round(pointsRightSleeve[3][1])), 10, (255, 0, 0), -1)

    # Display the frame with landmarks and overlay
    cv2.imshow('Webcam Feed with Overlay', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()