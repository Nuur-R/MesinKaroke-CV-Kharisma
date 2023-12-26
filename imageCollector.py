import cv2
from pytube import YouTube
import os
import time

def download_youtube_video(url, save_path):
    yt = YouTube(url)
    video_title = yt.title
    video_path = os.path.join(save_path, f"{video_title}.mp4")

    if os.path.exists(video_path):
        print(f"Video '{video_title}' already exists. Skipping download.")
    else:
        ys = yt.streams.get_highest_resolution()
        ys.download(save_path)
        print(f"Downloaded video '{video_title}' to {video_path}")


def extract_frames(video_path, output_folder, interval=3):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / frame_rate

    current_time = 0
    frame_number = 0

    while current_time < total_duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()

        if ret:
            output_file = f"{output_folder}/{YouTube(youtube_url).title}_frame_{frame_number}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Captured frame {frame_number} at {current_time:.2f} seconds")
            frame_number += 1

        current_time += interval

    cap.release()

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=63z9PLNmEqQ"
    save_path = "./save"
    output_folder = "./output"
    
    # Download video from YouTube
    download_youtube_video(youtube_url, save_path)

    # Extract frames from the downloaded video
    extract_frames(f"{save_path}/{YouTube(youtube_url).title}.mp4", output_folder)
