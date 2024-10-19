import cv2
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def minustuple(ta, tb):
    return (ta[0]-tb[0],ta[1]-tb[1])
def resize(NE, NW, SE, SW, clothesimage):
    maxX = max(abs(NE[0]-NW[0]), abs(SE[0]-SW[0]))
    width = round(maxX)
    maxY = max(abs(SW[1]-NW[1]), abs(SE[1]-NE[1]))
    height = round(maxY)
    #print(clothesimage.shape, NE, NW, SE, SW, maxX, maxY)
    img = cv2.resize(clothesimage, (round(maxX), round(maxY)))
    pivot = (round(min(NW[0], SW[0])), round(min(NW[1], NE[1])))
    NE = minustuple(NE, pivot)
    NW = minustuple(NW, pivot)
    SE = minustuple(SE, pivot)
    SW = minustuple(SW, pivot)
    control_points_displacements = {
        (0,0): (NW[0], NW[1]),
        (width, 0): (NE[0]-width, NE[1]),
        (0, height): (SW[0], SW[1]-height),
        (width, height): (SE[0]-width, SE[1]-height),
    }
    control_points = np.array(list(control_points_displacements.keys()))
    displacements = np.array(list(control_points_displacements.values()))

    new_control_points = control_points + displacements

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.array(list(zip(x.flatten(), y.flatten())))

    new_points = griddata(new_control_points, control_points, grid_points, method='cubic')

    map_x = new_points[:, 0].reshape(height, width).astype(np.float32)
    map_y = new_points[:, 1].reshape(height, width).astype(np.float32)

    warped_image = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    mask = np.zeros((height, width), dtype=np.uint8)

    hull = cv2.convexHull(new_control_points.astype(np.int32))

    cv2.fillPoly(mask, [hull], 255)

    warped_image_rgba = cv2.cvtColor(warped_image, cv2.COLOR_BGR2BGRA)

    warped_image_rgba[:, :, 3] = mask

    cv2.imwrite('warped_image.png', warped_image_rgba)
    #original_x, original_y = control_points[:, 0], control_points[:, 1]
    # new_x, new_y = new_control_points[:, 0], new_control_points[:, 1]
    # plt.rcParams['figure.figsize'] = [10, 10]
    # plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB), extent=[pivot[0], pivot[0] + warped_image.shape[1], pivot[1], pivot[1] + warped_image.shape[0]])
    
    # plt.imshow(cv2.cvtColor(warped_image_rgba, cv2.COLOR_BGRA2RGBA), extent=[pivot[0], pivot[0] + warped_image_rgba.shape[1], pivot[1], pivot[1] + warped_image_rgba.shape[0]])
    
    # plt.scatter(original_x, original_y, color='red', label='Original Points')
    # plt.scatter(new_x, new_y, color='blue', label='Displaced Points')

    # plt.legend(loc='upper right')
    # plt.title('title')
    
    # plt.show()

    return warped_image_rgba, pivot

def overlay_image(base_image, overlay_image, pivot):
    # Get dimensions of the overlay image
    overlay_height, overlay_width, _ = overlay_image.shape
    
    # If the base image does not have an alpha channel, add one (make it RGBA)
    if base_image.shape[2] == 3:
        base_image = np.dstack([base_image, np.ones((base_image.shape[0], base_image.shape[1]), dtype=np.uint8) * 255])
    
    # Extract the position where to place the overlay image (pivot point for bottom-left corner)
    
    pivot_x, pivot_y = pivot
    
    # Calculate the top-left corner of where the overlay will be placed on the base image
    top_left_x = pivot_x
    top_left_y = pivot_y - overlay_height  # Since pivot is bottom-left, subtract the height

    # Adjust the overlay image if it goes out of bounds (crop it to fit within base image)
    if top_left_x < 0:  # Overlay extends left of the base image
        overlay_image = overlay_image[:, -top_left_x:]  # Crop the left side of the overlay
        overlay_width = overlay_image.shape[1]
        top_left_x = 0  # Adjust the top-left x to 0

    if top_left_y < 0:  # Overlay extends above the base image
        overlay_image = overlay_image[-top_left_y:, :]  # Crop the top part of the overlay
        overlay_height = overlay_image.shape[0]
        top_left_y = 0  # Adjust the top-left y to 0

    if top_left_x + overlay_width > base_image.shape[1]:  # Overlay extends right of the base image
        overlay_image = overlay_image[:, :base_image.shape[1] - top_left_x]  # Crop the right side of the overlay
        overlay_width = overlay_image.shape[1]

    if top_left_y + overlay_height > base_image.shape[0]:  # Overlay extends below the base image
        overlay_image = overlay_image[:base_image.shape[0] - top_left_y, :]  # Crop the bottom of the overlay
        overlay_height = overlay_image.shape[0]

    # Extract the regions of interest (ROI) from the base image where the overlay will be placed
    roi = base_image[top_left_y:top_left_y + overlay_height, top_left_x:top_left_x + overlay_width]

    # if overlay_image == None:
    #     return base_image

    # Split the overlay into its RGBA channels
    overlay_b, overlay_g, overlay_r, overlay_a = cv2.split(overlay_image)

    # Normalize the alpha channel to be in range [0, 1]
    alpha_overlay = overlay_a / 255.0
    alpha_base = 1.0 - alpha_overlay

    # Perform blending on each channel (BGR)
    for c in range(3):  # Loop over B, G, R channels
        roi[:, :, c] = (alpha_overlay * overlay_image[:, :, c] + alpha_base * roi[:, :, c])

    # Apply the overlay's alpha channel to the base image's alpha channel
    roi[:, :, 3] = (alpha_overlay * 255 + alpha_base * roi[:, :, 3])

    # Replace the region of interest on the base image with the blended result
    base_image[top_left_y:top_left_y + overlay_height, top_left_x:top_left_x + overlay_width] = roi

    return base_image
    # overlay_height, overlay_width, _ = overlay_image.shape
    # pivot_x, pivot_y = pivot
    # top_left_x = pivot_x
    # top_left_y = pivot_y - overlay_height
    # if top_left_x < 0 or top_left_y < 0 or top_left_x + overlay_width >  base_image.shape[1] or top_left_y + overlay_height > base_image.shape[0]:
    #     print("Overlay out of bounds")
    #     return base_image
    
    # roi = base_image[top_left_y:top_left_y+round(overlay_height), top_left_x:top_left_x+round(overlay_width)]
    # overlay_b, overlay_g, overlay_r, overlay_a = cv2.split(overlay_image)
    # alpha_overlay = overlay_a / 255.0
    # alpha_base = 1.0 - alpha_overlay
    # for c in range(0, 3):
    #     roi[:, :, c] = (alpha_overlay * overlay_image[:, :, c] + alpha_base * roi[:, :, c])
    
    # base_image[top_left_y:top_left_y+overlay_height, top_left_x:top_left_x+overlay_width] = roi

    # return base_image

# resize((50, 50), (150, 60), (55, 220), (145, 190))
# resize((150, 60), (50, 50), (145, 190), (55, 220))