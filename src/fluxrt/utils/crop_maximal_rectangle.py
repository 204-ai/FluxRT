import numpy as np
import cv2


def crop_maximal_rectangle(image: np.ndarray, target_height: int, target_width: int, area_downscale: bool = True) -> np.ndarray:
    """
    Crops maximal rectangle with target aspect ratio centered by remaining axis,
    then resizes cropped tile to target shape. Output is not "stretched" like with common resize.

    Args:
        image: numpy ndarray, the input image where rectangle would be cropped.
        target_height: int, height of output image
        target_width: int, width of output image

    Returns:
        image with shape (target_height, target_width, 3)
    """
    aspect_ratio = target_width / target_height

    input_image_height, input_image_width = image.shape[:2]
    input_aspect_ratio = input_image_width / input_image_height

    if aspect_ratio > input_aspect_ratio:
        crop_height = int(round(input_image_width / aspect_ratio))
        crop_height = max(1, min(input_image_height, crop_height))
        crop_width = input_image_width
    else:
        crop_width = int(round(input_image_height * aspect_ratio))
        crop_width = max(1, min(input_image_width, crop_width))  
        crop_height = input_image_height

    start_x = (input_image_width - crop_width) // 2
    start_y = (input_image_height - crop_height) // 2

    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    # INTER_AREA was passed positionally (the `dst` slot), so this always ran
    # INTER_LINEAR — which aliases on a downscale. AREA to shrink, LINEAR to grow;
    # area_downscale=False restores the old always-LINEAR behavior for A/B.
    shrinking = crop_width > target_width or crop_height > target_height
    interpolation = cv2.INTER_AREA if shrinking and area_downscale else cv2.INTER_LINEAR
    image = cv2.resize(cropped_image, (target_width, target_height), interpolation=interpolation)

    return image