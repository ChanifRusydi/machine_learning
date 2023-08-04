import cv2
import numpy as np  

def image_stitching(image1, image2):
expos_comp = cv2.detail.ExposureCompensator_GAIN_BLOCKS
ba_cost_func = cv2.detail_BundleAdjusterRay
features_finder = cv2.ORB_create()
seam_finder = cv2.detail.SeamFinder_NO
estimator = cv2.detail_HomographyBasedEstimator
warp_type  = 'plane'
wave_correct = 'horiz'
blend_type = 'multiband'
blend_strength = 5

matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.3)
compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp)

match_conf = 0.65
work_megapix = 0.6
seam_megapix = 0.1
compose_megapix = -1
conf_thresh = 1.0
ba_refine_mask = 'xxxxx'
wave_correct = wave_correct

seam_work_aspect = 1
full_img_sizes = []
features = []
images = []
is_work_scale_set = False
is_seam_scale_set = False
is_compose_scale_set = False
image_names = [image1, image2]
for name in image_names:
    full_img_sizes.append((name.shape[1], name.shape[0]))
    # this can be simplified
    if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
    else:
        if is_work_scale_set is False:
            work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_work_scale_set = True
        img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
    if is_seam_scale_set is False:
        if seam_megapix > 0:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
        else:
            seam_scale = 1.0
        seam_work_aspect = seam_scale / work_scale
        is_seam_scale_set = True
    img_feat = cv.detail.computeImageFeatures2(features_finder, img)

