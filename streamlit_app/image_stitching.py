import cv2
import numpy as np
import platform

# if success status = 0 and return image result
def image_stitching(image1, image2):
    expos_comp = cv2.detail.ExposureCompensator_GAIN_BLOCKS
    ba_cost_func = cv2.detail_BundleAdjusterRay()
    
    features_finder_type = 'AKAZE'
    if features_finder_type == 'AKAZE':
        features_finder = cv2.AKAZE_create()
        match_conf = 0.65
    elif features_finder_type == 'ORB':
        features_finder = cv2.ORB_create()
        match_conf = 0.3
    elif features_finder_type == 'BRISK':
        features_finder = cv2.BRISK_create()
        match_conf = 0.65
    
    print('features finder', features_finder)
    seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
    estimator = cv2.detail_AffineBasedEstimator()
    warp_type  = 'plane'
    wave_correct = 'plane'
    blend_type = 'multiband'
    blend_strength = 15     #overlap
    if platform.system() == 'Linux':
        try_cuda = True
    else:
        try_cuda = False
    matcher = cv2.detail.BestOf2NearestMatcher(try_cuda, match_conf=match_conf,num_matches_thresh1= 6,num_matches_thresh2= 6)
    compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp)
   
    work_megapix = 2
    seam_megapix = 0.2
    compose_megapix = -1
    conf_thresh = 1.0
    ba_refine_mask = 'xxxxx'
    wave_correct = cv2.detail.WAVE_CORRECT_HORIZ

    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    image_names = [image1, image2]

    if image1 is None or image2 is None:
        status = -1
        return status, None
    
    for name in image_names:
        full_img = name
        full_img_sizes.append((name.shape[1], name.shape[0]))
        print(full_img_sizes)
        # this can be simplified
        if work_megapix < 0:
                img = full_img
                work_scale = 1
                is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set =    True
            img = cv2.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_LINEAR_EXACT)
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
        img = cv2.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        images.append(img)
    try:
        p = matcher.apply2(features)
        matcher.collectGarbage()
    except:
        return -1, None

    indices = cv2.detail.leaveBiggestComponent(features, p, 0.3)
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    print(len(indices))

    for i in range(len(indices)):
        img_names_subset.append(image_names[indices[i]])
        img_subset.append(images[indices[i]])
        full_img_sizes_subset.append(full_img_sizes[indices[i]])
        # indices[i][0] = len(img_names_subset) - 1
        # indices[i][1] = len(img_names_subset) - 1

    images = img_subset
    image_names = img_names_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(img_names_subset)

    b,cameras = estimator.apply(features, p, None)
    if not b:
        return -1, None
        # print("Homography estimation failed.")
        # exit()

    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = ba_cost_func
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
    print('type of p', type(p))
    print('type of cameras', type(cameras))
    print('type of features', type(features))
    try:
        b, cameras = adjuster.apply(features, p, cameras)
    except:
        return -1, None
        # print("Camera parameters adjusting failed.")
        # exit()
    if not b:
        return -1, None
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
        cv2.detail.waveCorrect(rmats, wave_correct)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]

    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []

    for i in range(0, num_images):
        um = cv2.UMat(255 * np.ones(images[i].shape[:2], dtype=np.uint8))
        masks.append(um)

    warper = cv2.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0][0] *= swa
        K[0][2] *= swa
        K[1][1] *= swa
        K[1][2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        corners.append(corner) 
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_REPLICATE)
        masks_warped.append(mask_wp.get())

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)
    seam_finder = seam_finder.find(images_warped_f, corners, masks_warped)
    compose_scale = 1
    corners = []
    sizes = []
    blenders = None
    timelapser = None

    for idx, name in enumerate(image_names):
        full_img = name
        if compose_megapix > 0:
            compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
        if is_compose_scale_set is False:
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(image_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])
                
        resize_factor = abs(compose_scale - 1) 
        
        if resize_factor > 1e-1:
            
            img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            print('img is full img')
            img = full_img
        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)

        mask = 255 * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_REFLECT_101)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
       
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv2.dilate(masks_warped[idx], None)
        seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
        mask_warped = cv2.bitwise_and(seam_mask, mask_warped)
        #blender
        if blenders is None:
            blenders = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            overlap_point1 = -(int(image1.shape[0]*100-blend_strength/100)) , image1.shape[1]
            overlap_point2 = [int(image2.shape[0]*100-blend_strength/100), image2.shape[1]]
            overlap_size1 = [image1.shape[0]/ blend_strength, image1.shape[1]]
            overlap_size2 = [image2.shape[0]/blend_strength, image2.shape[1]]
            # overlap_sz = cv2.detail.overlapRoi[)
            # print('overlap size', overlap_sz)
            dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
            
            blend_width = image1.shape[1] * blend_strength / 100
            # blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            
            if blend_width < 1:
                blenders = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            elif blend_type == 'multiband':
                blenders = cv2.detail_MultiBandBlender()
                blenders.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
            elif blend_type == 'feather':
                blenders = cv2.detail_FeatherBlender()
                blenders.setSharpness(1. / blend_width)
            blenders.prepare(dst_sz)

        blenders.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])
    crop_width = image1.shape[1] + image2.shape[1]
    crop_height = min(image1.shape[0], image2.shape[0])
    print('crop width', crop_width)
    print('crop height', crop_height)
    mid_x, mid_y = int(crop_width/2), int(crop_height/2)

    result = None
    result_mask = None
    result, result_mask = blenders.blend(result, result_mask)
    # cropped_result = result[0:crop_height, 0:crop_width]
    result = result.astype(np.int16)
    status = 0
    # return status, cropped_result
    return status, result

if __name__ == "__main__":
    image1 = cv2.imread('image1_60_left.jpg')
    image2 = cv2.imread('image1_60_right.jpg')
    cv2.imshow('image1', image1)
    cv2.imshow('image2', image2)
    cv2.waitKey(0)
    status, image_result = image_stitching(image1=image1, image2=image2)
    print('status',status, image_result.shape)
    cv2.imshow('image_result', image_result)
    cv2.waitKey(0)