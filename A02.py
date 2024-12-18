import cv2
import numpy as np
from enum import Enum 

class OPTICAL_FLOW:    
    HORN_SHUNCK = "horn_shunck" 
    LUCAS_KANADE = "lucas_kanade"
    
def compute_video_derivatives(video_frames, size):
    
    if size == 2:
        kfx = np.array([[-1, 1],
                        [-1, 1]])
        
        kfy = np.array([[-1,-1],
                        [1, 1]])
        
        kft1 = np.array([[-1, -1],
                         [-1, -1]])
                        
        kft2 = np.array([[1, 1],
                         [1 , 1]])
        
    elif size == 3:
        kfx = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
        
        kfy = np.array([[-1,-2,-1],
                        [0, 0, 0],
                        [1, 2, 1]])
        
        kft1 = np.array([[-1, -2, -1],
                         [-2, -4, -2],
                        [ -1, -2, -1]])
        
        kft2 = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])
    else:
        return None

    previous_frame = None
    
    all_fx = []
    all_fy = []
    all_ft = []
    
    for frame in video_frames:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        
        if previous_frame is None:
            previous_frame = grayscale.copy()
        
        fx = (cv2.filter2D(previous_frame, cv2.CV_64F, kfx) + cv2.filter2D(grayscale, cv2.CV_64F, kfx))
        fy = (cv2.filter2D(previous_frame, cv2.CV_64F, kfy) + cv2.filter2D(grayscale, cv2.CV_64F, kfy))
        ft = (cv2.filter2D(previous_frame, cv2.CV_64F, kft1) + cv2.filter2D(grayscale, cv2.CV_64F, kft2))
        
        if size == 2:
            fx /= 4.0
            fy /= 4.0
            ft /= 4.0
        elif size == 3:
            fx /= 8.0
            fy /= 8.0
            ft /= 16.0
        
        all_fx.append(fx)
        all_fy.append(fy)
        all_ft.append(ft)
        
        previous_frame = grayscale.copy()
    
    return all_fx, all_fy, all_ft


#commented this one cuz its a pain and maybe ill figure out where i messed up
#tried to use prof exercises to write it
def compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, weight=1.0):
    # initialize flow vectors
    u = np.zeros_like(fx, dtype=np.float64)
    v = np.zeros_like(fx, dtype=np.float64)

    #define the filter
    filter = np.array([[0, 0.25, 0],
                       [0.25, 0, 0.25],
                       [0, 0.25, 0]], dtype=np.float64)

    #partial derivative filters
    kfx = np.array([[-1, 1]], dtype=np.float64)
    kfy = np.array([[-1], [1]], dtype=np.float64)

    #initialize iterations and set final cost to infinity so it is greater than max error
    iterations = 0
    finalcost = np.inf
    lamb = weight

    #keeps iterating while iterations is less than maxiter and final cost is greater than maxerror
    while iterations < max_iter and finalcost > max_error:
        #applies filter to get local averages
        ulocal = cv2.filter2D(u, cv2.CV_64F, filter)
        vlocal = cv2.filter2D(v, cv2.CV_64F, filter)
        #updates flow vectors
        numerator = fx * ulocal + fy * vlocal + ft
        denominator = lamb + fx**2 + fy**2
        divided = numerator / denominator
        u = ulocal - (fx * divided)
        v = vlocal - (fy * divided)
        #gets spatial gradients of u and v
        ux = cv2.filter2D(u, cv2.CV_64F, kfx)
        uy = cv2.filter2D(u, cv2.CV_64F, kfy)
        vx = cv2.filter2D(v, cv2.CV_64F, kfx)
        vy = cv2.filter2D(v, cv2.CV_64F, kfy)
        #calculates error hopefully?
        data = np.mean((fx * u + fy * v + ft)**2)
        smoothness = lamb * np.mean(ux*uy + uy*uy + vx*vx + vy*vy)
        error = data + smoothness
        finalcost = error
        #updates iteration
        iterations += 1
    #fixes shape error? might be wrong, grabbed from prof exercise
    zeros = np.zeros_like(u)
    flow = np.stack((u, v, zeros), axis=-1)

    return flow, finalcost, iterations


def compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK, max_iter=10, max_error=1e-4, horn_weight=1.0, kanade_win_size=19):
    if method == OPTICAL_FLOW.HORN_SHUNCK:
        size = 2
        all_fx, all_fy, all_ft = compute_video_derivatives(video_frames, size)
        
        flows = []
        #think theyre all the same size
        frames = len(all_fx)

        for x in range(frames):
            fx = all_fx[x]
            fy = all_fy[x]
            ft = all_ft[x]
        
            flow, _, _ = compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, horn_weight)
            flows.append(flow)

        return flows




def main():
    return
    
    
if __name__ == "__main__":     
    main()
    