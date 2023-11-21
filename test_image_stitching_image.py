import cv2
import time
from streamlit_app.image_stitching import image_stitching

import logging

logging.basicConfig(level=logging.INFO, filename="logfile.txt", filemode="a", format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
image_name1 = "images/image1_60_left.jpg"
image_name2 = "images/image1_60_right.jpg"
# Load images
image1 = cv2.imread(image_name1)
image2 = cv2.imread(image_name2)
logger.info(image_name1)
logger.info(image_name2)

# Stitch images
start_time = time.time()
status, stitched_image = image_stitching(image1, image2)
if status == -1:
    logger.error("Images are not stitched")
    print("Images are not stitched")
    stitched_image = cv2.hconcat([image1, image2])
    cv2.imwrite("images/hconcat_image.jpg", stitched_image)
elif stitched_image is not None:
    logger.info("Images are stitched")
    cv2.imwrite("images/stitched_image.jpg", stitched_image)
end_time = time.time()
print("Time taken to stitch images: ", end_time - start_time)
logger.info("Time taken to stitch images: {}".format(end_time - start_time))