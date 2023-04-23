import argparse

import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

def calculate_optical_flow_between_frames(frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
    """Calculate optical flow between two frames.

    Args:
        frame_1: The first frame as a NumPy array.
        frame_2: The second frame as a NumPy array.

    Returns:
        The optical flow as a NumPy array.
    """
    frame_1_gray, frame_2_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.calcOpticalFlowFarneback(frame_1_gray,
                                                frame_2_gray,
                                                None,
                                                pyr_scale = 0.5,
                                                levels = 3,
                                                winsize = 10,
                                                iterations = 3,
                                                poly_n = 5,
                                                poly_sigma = 1.1,
                                                flags = 0
                                                )
    return optical_flow


def generate_frames_in_between(frame_1: np.ndarray, frame_2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Generate intermediate frames between two frames using optical flow.

    Args:
        frame_1: The first frame as a NumPy array.
        frame_2: The second frame as a NumPy array.
        num_frames: The number of intermediate frames to generate.

    Returns:
        A list of generated frames as NumPy arrays.
    """
    resultant_frames = []
    resultant_frames.append(frame_1)
    optical_flow = calculate_optical_flow_between_frames(frame_1, frame_2)
    h, w = optical_flow.shape[:2]
    for frame_num in range(1, num_frames+1):
        alpha = frame_num / (num_frames + 1)
        flow =  -1 * alpha * optical_flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        interpolated_frame = cv2.remap(frame_1, flow, None, cv2.INTER_LINEAR)
        resultant_frames.append(interpolated_frame)
    return resultant_frames


def main(video_name: str, num_frames: int, video_type: str, output_path: str, output_height: int) -> None:
    """
    Generate a slow motion or higher fps video from a given video.

    Args:
        video_name (str): Path to the input video.
        num_frames (int): Number of frames to generate between every two frames.
        video_type (str): Type of video to generate, either 'slow' or 'butter'.
        output_path (str): Path to the output video
        output_height (int): output video height (default=300), output video width is scaled maintaining the aspect ratio. 

    Returns:
        None.

    """
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("input video fps = {}, width = {}, height = {}, total number of frames = {}".format(fps, width, height, video_num_frames))
    if video_type == 'high-fps':
        fps = fps * (num_frames + 1)
    aspect = width / height
    new_height = output_height
    new_width = int(aspect * new_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    num_frames_written = 0
    is_frame_1_available, frame_1 = cap.read()
    frame_1 = cv2.resize(frame_1, (new_width, new_height))
    with tqdm(total=video_num_frames) as pbar:
        while is_frame_1_available:
            is_frame_2_available, frame_2 = cap.read()
            if not is_frame_2_available:
                break
            frame_2 = cv2.resize(frame_2, (new_width, new_height))
            new_frames = generate_frames_in_between(frame_1, frame_2, num_frames)
            for new_frame in new_frames:
                out.write(new_frame)
                num_frames_written = num_frames_written + 1
            frame_1 = frame_2
            pbar.update(1)
        out.write(frame_1)
        pbar.update(1)
        out.release()
    print("output video fps = {}, width = {}, height = {}, total number of frames = {}".format(fps, new_width, new_height, num_frames_written))
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', required=True, help="video path")
    parser.add_argument('--num_frames', '-n', required=True, type=int, help="number of frames to generate between every two frames")
    parser.add_argument('--type', choices=['slow-motion', 'high-fps'], default='slow', help="whether to generate a slow motion video or genreate a higher fps video")
    parser.add_argument('--output_path', default='output.mp4', help="output path to the video")
    parser.add_argument('--output_height', default=300, type=int, help="scale the height of the width to these many pixels while maintaing the aspect ratio of the video")
    args = parser.parse_args()
    main(args.video, args.num_frames, args.type, args.output_path, args.output_height)

