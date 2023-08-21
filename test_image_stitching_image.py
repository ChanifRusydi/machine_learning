import cv2
import time
from streamlit_app.image_stitching import image_stitching

import logging

logging.basicConfig(level=logging.INFO, filename="logs/test_image_stitching_image.log", filemode="a", format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
image_name1 = "images/1.jpg"
image_name2 = "images/2.jpg"
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
elif stitched_image is not None:
    logger.info("Images are stitched")
    cv2.imwrite("images/stitched_image.jpg", stitched_image)
end_time = time.time()
print("Time taken to stitch images: ", end_time - start_time)
logger.info("Time taken to stitch images: {}".format(end_time - start_time))