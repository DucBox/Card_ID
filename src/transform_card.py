import cv2
import numpy as np

def perspective_transform(image, corners, output_size=(800, 500)):
    """
    Thực hiện transform perspective dựa trên 4 góc nhận diện được.
    """
    print("[INFO] Performing perspective transform...")

    required_corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    if not all(corner in corners for corner in required_corners):
        print("[ERROR] Missing corners for perspective transform!")
        return None

    # Lấy tọa độ góc
    src_points = np.array([
        corners["top_left"],
        corners["top_right"],
        corners["bottom_left"],
        corners["bottom_right"]
    ], dtype=np.float32)

    dst_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [0, output_size[1] - 1],
        [output_size[0] - 1, output_size[1] - 1]
    ], dtype=np.float32)

    # Tính ma trận transform
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(image, M, output_size)

    print("[INFO] Perspective transform complete.")
    return transformed_image
