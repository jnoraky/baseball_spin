import numpy as np
import cv2
import matplotlib.pyplot as plt

from detect_baseball import process_data
from edge_detect import get_seam_pix
from estimate_orientation import estimate_orientation

Rmat = np.load('rotations.npy')
Rmat2 = np.load('rotations2.npy')
xyz_rotated = np.load('xyz_rotated.npy')
cost_matrix = np.load('cost_matrx.npy')


def best_rotation(xyz,edge_vec,r_vec,cent_vec,start,Rmat2,numIter):
   # edge, r, is a 
   minCost = np.inf
   bestR = np.eye(3)

   numEdges = len(edge_vec)
   numRot = numIter
   
   xx_vec = []
   yy_vec = []
   for ii in range(numEdges):
      yy,xx = np.nonzero(edge_vec[ii])
      xx_vec.append(xx.flatten())
      yy_vec.append(yy.flatten())
   
   #cost_vec = np.zeros(numRot)
   for jj in range(numRot):
      R_tmp = Rmat2[3*jj:3*(jj+1),:]
      
      xyz_tmp = xyz.copy()
      #count = start
      #while count != 0:
      #   xyz_tmp = np.dot(R_tmp.transpose(),xyz_tmp);
      #   count -= 1


      tmpCost = 0
      
      for ii in range(start+1,start+4):#range(numEdges):
         xyz_tmp = np.dot(R_tmp,xyz_tmp)
         r = r_vec[ii]
         cx,cy = cent_vec[ii]
         #edge = edge_vec[ii]
   
         # Can pr
         nzindx = xyz_tmp[2,:] > 0
         x_data = r*xyz_tmp[0,nzindx]+cx
         y_data = r*xyz_tmp[1,nzindx]+cy
         num_nz = len(x_data)

         xx = xx_vec[ii]
         yy = yy_vec[ii]
         scale = len(xx)
        
         if scale != 0:
         
            dx = np.dot(np.ones((scale,1)),x_data[np.newaxis,:])-\
               np.dot(xx[:,np.newaxis],np.ones((1,num_nz))) 
            dx = dx**2
            dy = np.dot(np.ones((scale,1)),y_data[np.newaxis,:])-\
               np.dot(yy[:,np.newaxis],np.ones((1,num_nz))) 
            dy = dy**2
            err = (dx+dy)**0.5
            err = np.amin(err,axis=1)
         
            # Set a threshold
            if np.max(err) > 5e2:
               tmpCost = np.inf
               break
            tmpCost += 1.*np.sum(err)#/num_nz
         else:
            tmpCost += 0
         #xyz_tmp = np.dot(R_tmp,xyz_tmp)
        
      #cost_vec[jj] = tmpCost
      if tmpCost < minCost:
         minCost = tmpCost
         bestR = R_tmp.copy()
   
   return bestR,minCost#,cost_vec

if __name__ == '__main__':
   
   # Load the baseball frames and extract the baseballs
   path_to_frames = \
      '/Users/james/Desktop/Consulting/repo/baseball_spin/SpinTests/015'
   prefix = ''
   first_frame = 9

   im_vec,r_vec,cent_vec = process_data(path_to_frames,prefix,first_frame)
 
   edge_vec = []
   xyz_vec = []

   # Edge detection parameters
   filter_size = 5
   logo_thresh = 0
   min_size = 0
   lap_thresh = 0
   min_dist = 2
   visualize = 1

   for ii in range(len(im_vec)):
      img = im_vec[ii]
      r = r_vec[ii]
      cx,cy = cent_vec[ii]
      seam_pix = get_seam_pix(img,r,cx,cy,filter_size,logo_thresh,min_size,\
         lap_thresh)
   
      new_edge,xyz = estimate_orientation(seam_pix,r,cx,cy,min_dist,\
         xyz_rotated,cost_matrix)
      
      edge_vec.append(new_edge)
      xyz_vec.append(xyz)

      if visualize:
         plt.subplot(1,3,1)
         plt.imshow(img)
         plt.title('Image')
         plt.subplot(1,3,2)
         plt.imshow(seam_pix)
         plt.title('Raw Seam')
         plt.subplot(1,3,3)
         plt.imshow(new_edge)
         plt.title('Filtered Edge')
         plt.show()

   bestErr = np.Inf
   bestR = np.eye(3)

   numRot2Try = 10000
   fps = 240

   for start in range(len(xyz_vec)-3):
      xyz_tmp = xyz_vec[start].copy()
      R,minCost = best_rotation(xyz_tmp,edge_vec,r_vec,cent_vec,start,\
         Rmat2,numRot2Try)

      # We count the number of seam pixels, if it is too low,
      # we cannot reliably estimate spin or the axis
      numEdge = 0
      for kk in range(start,start+4):
         numEdge += np.sum(edge_vec[kk])
      minCost = minCost/numEdge 
      trR = np.trace(R)
      angle = np.arccos(0.5*(trR-1))
      spin_tmp = angle/(2*3.1415926)*fps*60
      print(minCost,',',numEdge,',',spin_tmp)
      if (numEdge > 0) and (minCost < bestErr):
         bestR = R.copy()
         bestErr = minCost

   if bestErr == np.inf:
      print('Image quality is too low. Estimation failed!')
   else:
      # Get Spin
      trR = np.trace(bestR)
      angle = np.arccos(0.5*(trR-1))
      spin = angle/(2*3.1415926)*fps*60
      
      # Get Axis
      axis = np.zeros(3)
      axis[0] = bestR[2,1]-bestR[1,2]
      axis[1] = bestR[0,2]-bestR[2,0]
      axis[2] = bestR[1,0]-bestR[0,1]

      print('Spin = {0} Axis = {1}'.format(spin,axis))
