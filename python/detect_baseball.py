import numpy as np
import cv2

def detect_ball(frame1,frame2):
   '''
   Detect baseball with frame differencing. NOTE: We are assuming black 
   background with no net
   
   INPUT: consecutive frames (frame1 and frame2)
   OUTPUT: ball_frame, radius, center
   '''

   diff = frame2-frame1
   numBallPix = np.sum(diff > 100)
   if numBallPix > 150:
      # get the background
      bg = (diff < 30).astype(np.uint8)
      strel = np.zeros((25,25),dtype=np.uint8)
      bg_erode = cv2.erode(bg,strel,iterations=1)
      frame2_tmp = frame2*(bg_erode == 0)

      # Get the stats on binary image of the ball
      # This includes bounding box, radius, etc...
      ballonly = (frame2_tmp > 0).astype(np.uint8)
      output = cv2.connectedComponentsWithStats(ballonly,8,cv2.CV_32S)
      numComponents = output[0]
      labels = output[1]
      stats = output[2]
      centroid = output[3]

      # Get the second largest
      sort_indx = np.argsort(stats[:,cv2.CC_STAT_AREA])
      ball_indx = sort_indx[-2]
      
      # get bounding box 
      x = stats[ball_indx,cv2.CC_STAT_LEFT]
      y = stats[ball_indx,cv2.CC_STAT_TOP]
      w = stats[ball_indx,cv2.CC_STAT_WIDTH]
      h = stats[ball_indx,cv2.CC_STAT_HEIGHT]

      ball_frame = frame2[y:y+h,x:x+w]
      r = 1.*(stats[ball_indx,cv2.CC_STAT_AREA]/3.1415926)**0.5
      cent = centroid[ball_indx,:]
      cent[0] -= x
      cent[1] -= y

      return ball_frame,r,cent

def process_data(path_to_frames,prefix,start_indx):
   '''
   Get a list of the ball images, radii, and centers
   '''
   im_vec = []
   r_vec = []
   cent_vec = []
   try:
      im = cv2.imread(path_to_frames+'/{0}{1}.png'.format(prefix,start_indx),0)
      i = start_indx+1
      while True:
         im_prev = im.copy()
         im = cv2.imread(path_to_frames+'/{0}{1}.png'.format(prefix,i),0)
         
         if len(im[0,:]) == 0:
            break
         ball_frame,r,cent = detect_ball(im_prev,im)
         im_vec.append(ball_frame)
         r_vec.append(r)
         cent_vec.append(cent)
         i += 1
   except:
      pass 
   return im_vec,r_vec,cent_vec 

if __name__ == '__main__':
   path_to_frames = '/Users/james/Desktop/Consulting/repo/baseball_spin/SpinTests/002'
   prefix = ''
   im_vec,r_vec,cent_vec = process_data(path_to_frames,prefix,8)
   import matplotlib.pyplot as plt
   for ii in range(len(im_vec)):
      plt.figure()
      plt.imshow(im_vec[ii])
      print(r_vec[ii])
      print(cent_vec[ii])
      plt.show()
