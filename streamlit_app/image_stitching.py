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
    img_feat = cv2.detail.computeImageFeatures2(features_finder, img)
    features.append(img_feat)
    # this need to be adjusted
    img = cv2.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
    images.append(img)

p = matcher.apply2(features)
matcher.collectGarbage()

indices = cv2.detail.leaveBiggestComponent(features, p, 0.3)
img_subset = []
img_names_subset = []
full_img_sizes_subset = []
for i in range(len(indices)):
    img_names_subset.append(image_names[indices[i][0]])
    img_subset.append(images[indices[i][0]])
    full_img_sizes_subset.append(full_img_sizes[indices[i][0]])
    indices[i][0] = len(img_names_subset) - 1
    indices[i][1] = len(img_names_subset) - 1
images = img_subset
image_names = img_names_subset
full_img_sizes = full_img_sizes_subset
num_images = len(img_names_subset)

b,cameras = estimator.apply(features, p, None)
if not b:
    print("Homography estimation failed.")
    exit()

for cam in cameras:
    cam.R = cam.R.astype(np.float32)

adjuster = ba_cost_func.create()
adjuster.setConfThresh(conf_thresh)
refine_mask = np.zeros((3, 3), np.uint8)
if ba_refine_mask[0] == 'x':
    refine_mask[0][0] = 1
if ba_refine_mask[1] == 'x':
    refine_mask[0][1] = 1
if ba_refine_mask[2] == 'x':
    refine_mask[0][2] = 1
if ba_refine_mask[3] == 'x':
    refine_mask[1][1] = 1
if ba_refine_mask[4] == 'x':
    refine_mask[1][2] = 1
adjuster.setRefinementMask(refine_mask)
b, cameras = adjuster.apply(features, p, cameras)
if not b:
    print("Camera parameters adjusting failed.")
    exit()

focals = []
for cam in cameras:
    focals.append(cam.focal)
focals.sort()
if len(focals) % 2 == 1:
    warped_image_scale = focals[len(focals) // 2]
else:
    warped_image_scale = focals[len(focals) // 2 - 1] + focals[len(focals) // 2]
if wave_correct is not None:
    rmats = []
    for cam in cameras:
        rmats.append(np.copy(cam.R))
    cv.detail.waveCorrect(rmats, wave_correct)
    for idx, cam in enumerate(cameras):
        cam.R = rmats[idx]
corners = []
masks_warped = []
images_warped = []
sizes = []
masks = []




