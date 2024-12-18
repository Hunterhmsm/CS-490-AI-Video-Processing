import cv2
import numpy as np

def track_doggo(video_frames, first_box):
    ymin, xmin, ymax, xmax = first_box
    initial_width = xmax - xmin
    initial_height = ymax - ymin
    
    #arbitrary increase
    track_window = (xmin, ymin, int(initial_width * 1.4), int(initial_height * 1.4))

    #region of interest
    roi = video_frames[0][ymin:ymax, xmin:xmax]

    #roi to hsv
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #mask to filter colors
    mask = cv2.inRange(hsv_roi, np.array((0, 30, 60)), np.array((40, 255, 255)))

    #histogram of hue and sat
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    #termination criteria for camshift
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

    bounding_boxes = [first_box]

    for frame_idx in range(1, len(video_frames)):
        frame = video_frames[frame_idx]

        #hsv convert
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #back projection
        back_proj = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

        #blur to smooth
        back_proj = cv2.GaussianBlur(back_proj, (9, 9), 0)

        #camshift things
        ret, track_window = cv2.CamShift(back_proj, track_window, term_crit)

        x, y, w, h = track_window

        #keep it around size of initial box
        w = min(max(w, int(initial_width * 0.8)), int(initial_width * 1.5))  
        h = min(max(h, int(initial_height * 0.8)), int(initial_height * 1.5))  

        #arbitrary expansion because it mostly tracks but isnt big enough
        y = max(0, y - int(h * 0.4))  #shift up
        x = max(0, x - int(w * 0.1))  #width increase
        w = int(w * 1.3)  #expand width to the right
        h = int(h * 1.3)  #expand height

        #box
        new_box = (max(0, y), max(0, x), min(y + h, frame.shape[0]), min(x + w, frame.shape[1]))
        #append
        bounding_boxes.append(new_box)

    return bounding_boxes
