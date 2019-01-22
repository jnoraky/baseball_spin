import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_circle_mask(h,w,r,cx,cy):
   mask = np.zeros((h,w))
   yy = range(h)
   xx = range(w)
   xx,yy = np.meshgrid(xx,yy)
   mask[ (1.*(xx-cx)**2+1.*(yy-cy)**2) <= 1.*r**2] =1
   return mask

def edge_detect(img,r,sigmac,sigmas):
   res = cv2.bilateralFilter(img,r,sigmac,sigmas) 
   res2 = cv2.Laplacian(res, cv2.CV_64F,ksize=r)
   return res2

def get_seam_pix(img,r,cx,cy,filter_size,logo_thresh,min_size,lap_thresh):
   h,w = img.shape
   if r == 0:
      seam_pix = np.zeros((h,w),'uint8')
      return seam_pix

   mask = get_circle_mask(h,w,0.8*r,cx,cy)
   edge = edge_detect(img,filter_size,75,75)

   edge_tmp = edge > lap_thresh
   edge_nb = edge_tmp*mask

   num,output,stats,_ = cv2.connectedComponentsWithStats(
      edge_nb.astype('uint8'),connectivity=4)
   sizes = stats[:,-1]
   sort_indx = np.argsort(sizes)
      
   best_indx = sort_indx[-2]
   largest_seg = sizes[best_indx]
      
   seam_pix_tmp = np.zeros((h,w))
   for ii in range(num):
      tmp = sizes[ii]
      if (tmp > min_size) and (tmp <= largest_seg):
          seam_pix_tmp += output==ii   
   mask = img > logo_thresh
   seam_pix = seam_pix_tmp*mask
   return seam_pix

if __name__ == '__main__':
   import detect_baseball
   path_to_frames = \
      '/Users/james/Desktop/Consulting/repo/baseball_spin/SpinTests/002'
   prefix = ''
   im_vec,r_vec,cent_vec = detect_baseball.process_data(path_to_frames,prefix,8)
  
   indx = 2
   img = im_vec[indx]
   r = r_vec[indx]
   cx,cy = cent_vec[indx]
   filter_size = 3
   logo_thresh = 0
   min_size = 0
   lap_thresh = 0
   seam_pix = get_seam_pix(img,r,cx,cy,filter_size,logo_thresh,min_size,lap_thresh)
   
   import matplotlib.pyplot as plt
   plt.figure()
   plt.imshow(img)
   plt.figure()
   plt.imshow(seam_pix)
   plt.show()
   
