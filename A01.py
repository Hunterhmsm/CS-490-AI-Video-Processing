import sys
import cv2
import os
import shutil
from pathlib import Path

def load_video_as_frames(video_filepath):
    capture = cv2.VideoCapture(video_filepath)
    
    frames = []
    
    if not capture.isOpened():
        print("Error opening capture.")
        return None
        
    while True:
        
        ret, frame = capture.read()
        
        if ret:
            frames.append(frame)
        else:
            break        
        
    capture.release()
            
    return frames

def compute_wait(fps):
    wait = int(1000.0/fps)
    return wait

def display_frames (all_frames, title, fps=30):
    wait =compute_wait(fps)
    
    for frame in all_frames:
        cv2.imshow(title, frame)
        
        key = cv2.waitKey(wait)
    
    cv2.destroyAllWindows()
    
def save_frames(all_frames, output_dir, basename, fps=30):
    folder = f"{basename}_{fps}"
    outpath = os.path.join(output_dir, folder)
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath)
    
    tracking = 0
    for frame in all_frames:
        
        filename = f"image_{tracking:07d}.png"
        
        fullpath = os.path.join(outpath, filename)
        
        cv2.imwrite(fullpath, frame)
        tracking += 1
        
def main():
    if (len(sys.argv)) < 3:
        print("Error len sys.argv")
        exit(1)
    input_video = sys.argv[1]
    output_directory = sys.argv[2]
    
    core_name = Path(input_video).stem
    
    all_frames = load_video_as_frames(input_video)
    if all_frames is None:
        print("Video frames error")
        exit(1)
    display_frames(all_frames,"Input Video", fps=30)
    save_frames(all_frames, output_directory, core_name, fps=30)
    
    
if __name__ == "__main__":     
    main()
    