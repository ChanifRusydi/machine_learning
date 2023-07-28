import cv2
import argparse
import time

import platform
import sys

def get_matcher():
    range_width=-1
    matcher_type = 'homography' # 'homography' or 'affine'
    match_conf = 0.3 # only for homography
    if matcher_type == 'affine':
        matcher= cv2.detail_AffineBestOf2NearestMatcher(False, False, match_conf)
    elif range_width== -1:
        matcher= cv2.detail.BestOf2NearestMatcher_create(False, match_conf)

def get_compensator(args):
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    return compensator

def image_stitching(args, img_names):
    if args.feature_finder == "AKAZE":
        feature_finder = cv2.AKAZE_create()
    elif args.feature_finder == "KAZE":    
        feature_finder=cv2.KAZE_create()
    elif args.feature_finder == "ORB":
        feature_finder=cv2.ORB_create()
    
    

def open_camera(index):
    cap = cv2.VideoCapture(index)
    return cap

def main(args):
    work_megapix = 0.6
    seam_megapix = 0.1
    features_type = 'AKAZE'
    estimator_type = 'homography'
    matcher_type = 'homography'
    feature_finder = cv2.AKAZE_create()
    match_conf = 0.3
    conf_thresh = 1.0
    ba_cost_func = 'ray'
    ba_refine_mask = 'xxxxx'
    


    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)

    if args.mode == 'image':
        # Read input image
        if args.image is None:
            print('Error: --image argument is required when --mode is set to "image"')
            return
        print(type(args.image), args.image)
        img1 = cv2.imread(args.image[0])
        img2 = cv2.imread(args.image[1])
        if args.image is None:
            print(f'Error: could not read image file "{args.image}"')
            return
        # Process input image
        # ...

    elif args.mode == 'camera':
        # Open camera
        cap = open_camera(0)
        cap1 = open_camera(1)

        if cap.isOpened():
            width = cap.get(3) # Frame Width
            height = cap.get(4) # Frame Height
            # Process camera input
            # ...

    elif args.mode == 'video':
        # Open video file
        if args.video is None:
            print('Error: --video argument is required when --mode is set to "video"')
            return 'Error: --video argument is required when --mode is set to "video"'
        print(args.video)
        video1 = cv2.VideoCapture(args.video[0])
        video2 = cv2.VideoCapture(args.video[1])
        video_status = []

        if not video1.isOpened() or not video2.isOpened():
            print(f'Error: could not open video file "{args.video}"')
            sys.exit()
        # Process video file
        # ...'
        fps1 = video1.get(cv2.CAP_PROP_FPS)
        frame_count1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
        duration1 = frame_count1/fps1
        height1, width1 = video1.get(4), video1.get(3)
        print(height1, width1)

        height2, width2 = video2.get(4), video2.get(3)
        fps2 = video2.get(cv2.CAP_PROP_FPS)
        frame_count2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
        duration2 = frame_count2/fps2
        print(height2, width2)

        print(f'fps1: {fps1}, frame_count1: {frame_count1}, duration1: {duration1}')
        print(f'fps2: {fps2}, frame_count2: {frame_count2}, duration2: {duration2}')

        frame_count_list = [frame_count1, frame_count2]

        for i in range(min(frame_count_list)):
            ret1, frame1 = video1.read()
            ret2, frame2 = video2.read()

            image_features1 =cv2.detail.computeImageFeatures2(feature_finder, frame1)
            image_features2 =cv2.detail.computeImageFeatures2(feature_finder, frame2)

            features1.append(image_features)
            frame1= cv2.resize(frame1, (int(width1), int(height1)))
            images1.append(frame1)

            features2.append(image_features)
            frame2= cv2.resize(frame2, (int(width2), int(height2)))

            matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.3)
            p = matcher.apply2(features)
            matcher.collectGarbage()

            indices = cv2.detail.leaveBiggestComponent(features, matches, 0.3)

            estimator = cv2.detail.HomographyBasedEstimator()
            b, cameras = estimator.apply(features, matches, indices)
            if not b:
                print('Homography estimation failed.')
                continue
            for camera in cameras:
                camera.R = camera.R.astype(np.float32)

            adjuster = cv2.detail.BundleAdjusterRay()
            adjuster.setConfThresh(1)
            refine_mask = np.zeros((3, 3), np.uint8)
            refine_mask[0,0] = 1
            refine_mask[0,1] = 1
            refine_mask[0,2] = 1
            refine_mask[1,1] = 1
            refine_mask[1,2] = 1
            adjuster.setRefinementMask(refine_mask)

            b, cameras = adjuster.apply(features, matches, cameras)
            if not b:
                print('Camera parameters adjusting failed.')
                continue
            focals = []
            for cam in cameras:
                focals.append(cam.focal)
            
            focals.sort()
            if len(focals) % 2 == 1:
                warped_image_scale = focals[len(focals) // 2]
            else:
                warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
            if wave_correct is not None:
                rmats = []
                for cam in cameras:
                    rmats.append(np.copy(cam.R))
                rmats = cv2.detail.waveCorrect(rmats, wave_correct)
                for idx, cam in enumerate(cameras):
                    cam.R = rmats[idx]
            corners = []
            masks_warped = []
            images_warped = []
            sizes = []
            masks = []
            for i in range(0, num_images):
                um = cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
                masks.append(um)

            warper = cv2.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
            for idx in range(0, num_images):
                K = cameras[idx].K().astype(np.float32)
                swa = seam_work_aspect
                K[0, 0] *= swa
                K[0, 2] *= swa
                K[1, 1] *= swa
                K[1, 2] *= swa
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

            compensator = get_compensator(args)
            compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

            seam_finder = SEAM_FIND_CHOICES[args.seam]
            masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
            compose_scale = 1
            corners = []
            sizes = []
            blender = None
            timelapser = None
            # https://github.com/opencv2/opencv2/blob/4.x/samples/cpp/stitching_detailed.cpp#L725 ?
            for idx, name in enumerate(img_names):
                full_img = cv2.imread(name)
                if not is_compose_scale_set:
                    if compose_megapix > 0:
                        compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_compose_scale_set = True
                    compose_work_aspect = compose_scale / work_scale
                    warped_image_scale *= compose_work_aspect
                    warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
                    for i in range(0, len(img_names)):
                        cameras[i].focal *= compose_work_aspect
                        cameras[i].ppx *= compose_work_aspect
                        cameras[i].ppy *= compose_work_aspect
                        sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                            int(round(full_img_sizes[i][1] * compose_scale)))
                        K = cameras[i].K().astype(np.float32)
                        roi = warper.warpRoi(sz, K, cameras[i].R)
                        corners.append(roi[0:2])
                        sizes.append(roi[2:4])
                if abs(compose_scale - 1) > 1e-1:
                    img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                    interpolation=cv2.INTER_LINEAR_EXACT)
                else:
                    img = full_img
                _img_size = (img.shape[1], img.shape[0])
                K = cameras[idx].K().astype(np.float32)
                corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
                mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
                p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
                compensator.apply(idx, corners[idx], image_warped, mask_warped)
                image_warped_s = image_warped.astype(np.int16)
                dilated_mask = cv2.dilate(masks_warped[idx], None)
                seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
                mask_warped = cv2.bitwise_and(seam_mask, mask_warped)

                if blender is None and not timelapse:
                    blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
                    dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
                    blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
                    if blend_width < 1:
                        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
                    elif blend_type == "multiband":
                        blender = cv2.detail_MultiBandBlender()
                        blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
                    elif blend_type == "feather":
                        blender = cv2.detail_FeatherBlender()
                        blender.setSharpness(1. / blend_width)
                    blender.prepare(dst_sz)
                elif timelapser is None and timelapse:
                    timelapser = cv2.detail.Timelapser_createDefault(timelapse_type)
                    timelapser.initialize(corners, sizes)
                if timelapse:
                    ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
                    timelapser.process(image_warped_s, ma_tones, corners[idx])
                    pos_s = img_names[idx].rfind("/")
                    if pos_s == -1:
                        fixed_file_name = "fixed_" + img_names[idx]
                    else:
                        fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
                    cv2.imwrite(fixed_file_name, timelapser.getDst())
                else:
                    blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])
            
            if not ret1 or not ret2:
                print('Error: could not read frame')
                sys.exit()
            side_by_side = cv2.hconcat([frame1, frame2])
            cv2.imshow('frame', side_by_side)
            if cv2.waitKey(1) == ord('q'):
                break
            # time.sleep(1/fps)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--mode', default='image', type=str, help='camera or video or image')
    parser.add_argument('--image', type=str,nargs='+',help='path to input image file')
    parser.add_argument('--camera', type=int, default=[0,1], help='camera index')
    parser.add_argument('--video', type=str,nargs='+', help='path to input video file')
    parser.add_argument('--feature_finder', type=str, default='AKAZE', help='AKAZE or KAZE or ORB')
    args = parser.parse_args()
    print(args.mode)
    print(args.image)
    print(args.camera)
    main(args)
# import cv2
# image1=cv2.imread('image1.jpg')
# image2=cv2.imread('image2.jpg')
# height1, width1 = image1.shape[:2]
# height2, width2 = image2.shape[:2]
# print(height1, width1, height2, width2)
# stitching=cv2.Stitcher.create()
# status,stitched_image=stitching.stitch((image1,image2))
# if status==0:
#     cv2.imshow('frame',stitched_image)
#     cv2.imwrite('stitched_image.jpg',stitched_image)
# else:
#     print(status)
