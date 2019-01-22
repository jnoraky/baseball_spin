import numpy as np
import cv2
import matplotlib.pyplot as plt

def rescale_edge(edge,r,cent):
   cx,cy = cent
   yy,xx = np.nonzero(edge)
   xx = xx.flatten()
   yy = yy.flatten()

   edge2 = np.zeros((49,49))
   xx_new = (xx-cx)*(20.0/r)+24
   yy_new = (yy-cy)*(20.0/r)+24
   for ii in range(len(xx_new)):
      xxtmp = int(round(xx_new[ii]))
      yytmp = int(round(yy_new[ii]))
      edge2[yytmp,xxtmp] = 1
   return edge2


def estimate_orientation(edge,r,cx,cy,min_dist,xyz_rotated,cost_matrix):
   edge2 = rescale_edge(edge,r,(cx,cy))
   n = len(cost_matrix[:,0])
   tmp = np.dot(np.ones((n,1)),edge2.flatten()[np.newaxis,:])
   res = np.sum(tmp*cost_matrix < min_dist,axis=1)
   indx = np.argmax(res)
   #minCost = res[indx]

   # Get the best orientation
   xyz = xyz_rotated[3*indx:3*(indx+1),:]
   # Retain only the edge fragments that coexist 
   
   num,output,stats,_ = cv2.connectedComponentsWithStats(
      edge.astype('uint8'),connectivity=4)
   
   new_edge = np.zeros(edge.shape) 
   h,w = edge.shape
   for ii in range(396):
      z = xyz[2,ii]
      if z > 0:
         x = int((r*xyz[0,ii]+cx))
         y = int((r*xyz[1,ii]+cy))
         if (x < 0) or (x >= w) or (y < 0) or (y >= h):
            continue
         label_tmp = output[y,x]
         if label_tmp != 0:
            new_edge += output == label_tmp
   new_edge = new_edge > 0
   return new_edge,xyz

if __name__ == '__main__':
   #Rmat = np.load('rotations.npy')
   #Rmat2 = np.load('rotations2.npy')
   xyz_rotated = np.load('xyz_rotated.npy')
   cost_matrix = np.load('cost_matrx.npy')


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
   min_size = 2
   lap_thresh = 0
   from edge_detect import get_seam_pix
   seam_pix = get_seam_pix(img,r,cx,cy,filter_size,logo_thresh,min_size,lap_thresh)
   
   min_dist = 1
   new_edge,xyz = estimate_orientation(seam_pix,r,cx,cy,min_dist,\
      xyz_rotated,cost_matrix)
 
   import matplotlib.pyplot as plt
   plt.figure()
   plt.imshow(img)
   plt.figure()
   plt.imshow(seam_pix)
   plt.figure()
   plt.imshow(new_edge)
   plt.show()
 
