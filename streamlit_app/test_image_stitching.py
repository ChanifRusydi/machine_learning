import cv2
import numpy as np  

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
        match_conf = 0.6
    
    print('features finder', features_finder)
    seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
    estimator = cv2.detail_HomographyBasedEstimator()
    warp_type  = 'plane'
    wave_correct = 'horizontal'
    blend_type = 'multiband'
    blend_strength = 10      #overlap

    # matcher = cv2.detail.BestOf2NearestMatcher(False, match_conf=match_conf,num_matches_thresh1= 6,num_matches_thresh2= 56)
    compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp)
   
    work_megapix = 1
    seam_megapix = 1
    compose_megapix = -1
    conf_thresh = 1.0
    ba_refine_mask = 'xxxxx'
    wave_correct = cv2.detail.WAVE_CORRECT_AUTO

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
            print(seam_scale)
            print(work_scale)
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
    print(type(cameras))
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
    try:
        b, cameras = adjuster.apply(features, p, cameras)
    except:
        return -1, None
    

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
    # problem ada di mask warped
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
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
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
                print('corners on compose', corners)
                sizes.append(roi[2:4])
                print('sizes on comopse', sizes)
        resize_factor = abs(compose_scale - 1) 
        print(resize_factor)
        if resize_factor > 1e-1:
            img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            print('img is full img')
            img = full_img
        cv2.imshow('img', img)
        cv2.waitKey(0)
        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        print('corner', corner)
        cv2.imshow('image_warped', image_warped)
        cv2.waitKey(0)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_REFLECT_101)
        print('type mask warped',type(mask_warped))
        print(mask_warped.shape)
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
            # corners = [-1152, -720]
            # sizes = [2304, 1440]
            dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
            print('destination size', dst_sz)
            blend_width = image1.shape[1] * blend_strength / 100
            # blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            print('blend_width', blend_width)
            if blend_width < 1:
                blenders = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            elif blend_type == 'multiband':
                blenders = cv2.detail_MultiBandBlender()
                blenders.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
            elif blend_type == 'feather':
                blenders = cv2.detail_FeatherBlender()
                blenders.setSharpness(1. / blend_width)
            blenders.prepare(dst_sz)
            
        cv2.imshow('image_warped', image_warped_s)
        cv2.imshow('mask_warped', mask_warped)
        cv2.imshow('corners', corners[idx])
        cv2.waitKey(0)
        blenders.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])

    result = None
    result_mask = None
    try :
        result, result_mask = blenders.blend(result, result_mask)
    except:
        return -1, None
    result = result.astype(np.uint8)
    status = 0
    return status, result

if __name__ == "__main__":
    # image1 = cv2.imread('Intersect_kiri.jpg')
    # image2 = cv2.imread('Intersect_kanan.jpg')
    # cv2.imshow('image1', image1)
    # cv2.imshow('image2', image2)
    # cv2.waitKey(0)
    # status, image_result = image_stitching(image1=image1, image2=image2)
    # print('status',status, image_result.shape)
    # cv2.imshow('image_result', image_result)
    # desired_size_width = 2*image1.shape[1]
    # desired_size_height = image1.shape[0]
    # image_result_desired = cv2.resize(src=image_result, dsize=(desired_size_width, desired_size_height), interpolation=cv2.INTER_LINEAR_EXACT)
    # image_result_desired = image_result[0:desired_size_height, 0:desired_size_width ]
    # cv2.imshow('image_result_desired', image_result_desired)
    # cv2.waitKey(0)
    camera1 = cv2.VideoCapture(0)
    camera2 = cv2.VideoCapture(1)
    camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret1,frame1 = camera1.read()
        ret2,frame2 = camera2.read()
        status, result = image_stitching(frame1, frame2)
        image_side = cv2.hconcat([frame1, frame2])
        cv2.imshow('image_side', image_side)
        if status == -1:
            print('image stitching failed')
        else:   
            cv2.imshow('result', result)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()
