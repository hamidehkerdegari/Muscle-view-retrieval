__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"


#This script is used to generate json file for every standard view in the video.


import cv2
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

global verbose
verbose = False


def msec_to_timestamp(current_timestamp):
    minutes = int(current_timestamp / 1000 / 60)
    seconds = int(np.floor(current_timestamp / 1000) % 60)
    ms = current_timestamp - np.floor(current_timestamp / 1000) * 1000
    current_contour_frame_time = '{:02d}:{:02d}:{:.3f}'.format(minutes, seconds, ms)
    return current_contour_frame_time


def is_annotation(cropped_image, th_RG_contour, th_GB_contour):
    # see if cropped image has annotation
    R = cropped_image[..., 2].astype(np.float)
    G = cropped_image[..., 1].astype(np.float)
    B = cropped_image[..., 0].astype(np.float)

    GB = (G - B)[np.abs(G - B) > 1]
    RG = (R - G)[np.abs(R - G) > 1]

    has_green_contour = bool(np.mean(RG) < th_RG_contour)
    has_blue_contour = bool(np.mean(GB) < th_GB_contour)
    # has_contour = bool(len(GB) > min_contour_pixels)
    # current_frame_has_no_contour = bool(len(GB) < max_no_contour_pixels)
    
    return has_green_contour, has_blue_contour


def process_video(video_file_path: str, bounds=(), th_d: int = 25, th_GB_contour: int = -5, th_RG_contour: int = -5, dsize: tuple = (256, 256), min_mask_size: int = 50000):
    """Select a bounding box and erase all content outside of it. Use the bounding
    box to define the content that you want to preserve.
    If no bounding box is given, then a UI will show on the first frame"""

    path, videoname = os.path.split(video_file_path)

    video_capture = cv2.VideoCapture(video_file_path)
    # Check if video opened successfully
    if not video_capture.isOpened():
        print('Error: Unable to read video! ' + video_file_path)
        raise


    # Read the first frame to define manually the bounding box
    success, image = video_capture.read()
    if len(bounds) == 0:
        if not success:
            print('Error: I could not read the video.')
            raise

        # Select ROI
        bounds = cv2.selectROI(image, showCrosshair=False, fromCenter=False)
        # go back to the first frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("Bounds are " + str(bounds))
        cv2.destroyAllWindows()

        bounds_detect = cv2.selectROI(image, showCrosshair=False, fromCenter=False)
        # go back to the first frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("Bounds detect are  " + str(bounds_detect))
        cv2.destroyAllWindows()
    elif not len(bounds) == 4:
        print("Bounds should be given as a tuple of 4 elements")
        exit(-1)

    # Now redo the bounds to fit a desired  aspect ratio
    if bounds[2] > bounds[3]*dsize[0]/dsize[1]:
        diff = bounds[2]-int(bounds[3]*dsize[0]/dsize[1])
        dif_l = int(diff/2)
        dif_r = diff-dif_l
        bounds = (bounds[0]+dif_l, bounds[1], bounds[2]-(dif_l + dif_r), bounds[3])
    else:
        diff = int(bounds[3]*dsize[0]/dsize[1]) - bounds[2]
        dif_l = int(diff / 2)
        dif_r = diff - dif_l
        bounds = (bounds[0], bounds[1]+dif_l , bounds[2], bounds[3]-(dif_l + dif_r) )

    print('Bounds adapted to {} to maintain aspect ratio'.format(bounds))


    video_annotation = {
        "video_file_name": videoname,
        "annotations": []
    }


    state = 1
    i = 0
    no_counter_imgs = 0
    while success:
        # Crop image
        cropped_image = image[int(bounds[1]):int(bounds[1] + bounds[3]), int(bounds[0]):int(bounds[0] + bounds[2]), :]

        has_green_contour, has_blue_contour = is_annotation(cropped_image, th_RG_contour, th_GB_contour)

        current_frame_idx = i
        current_frame_msec = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        current_frame_time = msec_to_timestamp(current_frame_msec)

        if state == 1 and has_blue_contour:
            state = 2
        if state == 2 and not has_blue_contour:
            state = 3
            start_frame_idx = current_frame_idx
            start_time = current_frame_time
        if state == 3 and has_green_contour:
            state = 4
            end_frame_idx = current_frame_idx
            end_time = current_frame_time

            if end_frame_idx - start_frame_idx < 150:
                state = 1
            else:

                print("start_time:", start_time, "end_time:", end_time, "start_frame_idx:", start_frame_idx, "end_frame_idx:", end_frame_idx, "no_counter_imgs:", no_counter_imgs)
                video_annotation["annotations"].append(dict(start_frame_idx=start_frame_idx,
                                                            end_frame_idx=end_frame_idx,
                                                            start_time=start_time,
                                                            end_time=end_time,
                                                            number_of_frames=end_frame_idx-start_frame_idx,
                                                            no_counter_imgs=no_counter_imgs))
                no_counter_imgs = 0

        if state == 4 and not has_green_contour:
            state = 1

        if state == 2 and has_blue_contour:
            no_counter_imgs += 1

        success, image = video_capture.read()
        i += 1

    return video_annotation


def run(video_file_path: str, out_path: str, bounds: tuple = (500, 200, 1100, 700), min_mask_size: int = 50000,
         dsize: tuple = (275, 175), th_GB_contour = -5, th_d: int = 1):

    if not os.path.isfile(video_file_path):
        print('Error: Video not found! {}'.format(video_file_path))
        raise

    video_annotation = process_video(video_file_path, bounds=bounds, th_GB_contour=th_GB_contour, th_d=th_d, min_mask_size=min_mask_size, dsize=dsize)

    with open(out_path, "w") as out_f:
        json.dump(video_annotation, out_f, indent=4)


# This section is used for generating json files for a dataset.
if __name__ == '__main__':
    path = 'Dataset path' # path to dataset
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.mp4'):
                video_file_path = os.path.join(root, name)
                print(video_file_path)
                out_path = video_file_path.replace(".mp4", ".json")
                print(out_path)
                run(video_file_path=video_file_path, out_path=out_path)


