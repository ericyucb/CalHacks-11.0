import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
WIDTH = 1280
HEIGHT = 720
# 3 image input
def clearify(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # Invert to get black square as foreground
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    output[:, :, :3] = image  # Copy original image color
    output[:, :, 3] = mask    # Use the mask for the alpha channel (transparency)
    ifroa = Image.fromarray(output)
    bbox = ifroa.getbbox()
    tra = ifroa.crop(bbox)
    output = np.asarray(tra)
    return output
def generateSegmentation(image_path: str):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    contour_points = contour[:, 0, :]
    bottom_threshold = 0.95 * bottommost[1]
    bottom_points = contour_points[contour_points[:, 1] > bottom_threshold]
    left_corner = bottom_points[np.argmin(bottom_points[:, 0])]
    right_corner = bottom_points[np.argmax(bottom_points[:, 0])]
    leftimg = clearify(image[:, :round(0.955*left_corner[0])])
    centerimg = clearify(image[:, round(0.955*left_corner[0]):round(1.03*right_corner[0])])
    rightimg = clearify(image[:, round(1.03*right_corner[0]):])
    cv2.imwrite(f'shirt_fitter/assets/{image_path.split('/')[-1]}_left.png', leftimg)
    cv2.imwrite(f'shirt_fitter/assets/{image_path.split('/')[-1]}_right.png', rightimg)
    cv2.imwrite(f'shirt_fitter/assets/{image_path.split('/')[-1]}_center.png', centerimg)
def convert(p): return [p[0] * WIDTH, p[1] * HEIGHT]
def lerp(a,b,t): return (1-t)*a+t*b
def llerp(a,b,t): return [(1-t)*a[0]+t*b[0],(1-t)*a[1]+t*b[1]]
def angb2vec(v1,v2): return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
def master(image1path: str, image2path: str, image3path: str, debug=False):
    selected_outfit = 1
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    overlay_images = [
        cv2.flip(cv2.flip(cv2.imread(f'shirt_fitter/assets/{image1path.split('/')[-1]}_center.png', cv2.IMREAD_UNCHANGED), 0), 1),
        cv2.flip(cv2.flip(cv2.imread(f'shirt_fitter/assets/{image2path.split('/')[-1]}_center.png', cv2.IMREAD_UNCHANGED), 0), 1),
        cv2.flip(cv2.flip(cv2.imread(f'shirt_fitter/assets/{image3path.split('/')[-1]}_center.png', cv2.IMREAD_UNCHANGED), 0), 1),
    ]
    right_sleeves = [
        cv2.imread(f'shirt_fitter/assets/{image1path.split('/')[-1]}_right.png', cv2.IMREAD_UNCHANGED),
        cv2.imread(f'shirt_fitter/assets/{image2path.split('/')[-1]}_right.png', cv2.IMREAD_UNCHANGED),
        cv2.imread(f'shirt_fitter/assets/{image3path.split('/')[-1]}_right.png', cv2.IMREAD_UNCHANGED),
    ]
    left_sleeves = [
        cv2.flip(cv2.imread(f'shirt_fitter/assets/{image1path.split('/')[-1]}_left.png', cv2.IMREAD_UNCHANGED), 1),
        cv2.flip(cv2.imread(f'shirt_fitter/assets/{image2path.split('/')[-1]}_left.png', cv2.IMREAD_UNCHANGED), 1),
        cv2.flip(cv2.imread(f'shirt_fitter/assets/{image3path.split('/')[-1]}_left.png', cv2.IMREAD_UNCHANGED), 1),
    ]
    overlay_imgs = map(lambda x: x[..., :3], overlay_images)
    right_slvs = map(lambda x: x[..., :3], right_sleeves)
    left_slvs = map(lambda x: x[..., :3], left_sleeves)
    overlay_alphas = map(lambda x: x[..., 3] / 255.0, overlay_images)
    right_alphas = map(lambda x: x[..., 3] / 255.0, right_sleeves)
    left_alphas = map(lambda x: x[..., 3] / 255.0, left_sleeves)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        overlay_img = overlay_imgs[selected_outfit]
        right_slv = right_slvs[selected_outfit]
        left_slv = left_slvs[selected_outfit]
        overlay_image = overlay_images[selected_outfit]
        right_sleeve = right_sleeves[selected_outfit]
        left_sleeve = left_sleeves[selected_outfit]
        overlay_alpha = overlay_alphas[selected_outfit]
        right_alpha = right_alphas[selected_outfit]
        left_alpha = left_alphas[selected_outfit]
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
            hipR = np.array(convert([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]))
            shouldR = np.array(convert([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]))
            elbowR = np.array(convert([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]))
            hipL = np.array(convert([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]))
            shouldL = np.array(convert([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]))
            elbowL = np.array(convert([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]))
            print(hipR, shouldR)
            theta = angb2vec(hipR-shouldR, elbowR-shouldR)
            theta2 = 1.5*np.pi-theta
            ltheta = angb2vec(hipL-shouldL, elbowL-shouldL)
            ltheta2 = ltheta-0.5*np.pi
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
            points = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for (x, y) in points]
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
            #cv2.circle(frame, (round(pointsRightSleeve[3][0]), round(pointsRightSleeve[3][1])), 10, (255, 0, 0), -1)

            if debug: 
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for point in points:
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)
                for i, point in enumerate(pointsLeftSleeve):
                    cv2.circle(frame, (round(point[0]), round(point[1])), 10, (255,0,0), -1)
                for i, point in enumerate(pointsRightSleeve):
                    cv2.circle(frame, (round(point[0]), round(point[1])), 10, (255,0,0), -1)
        cv2.imshow("Live Testing Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

