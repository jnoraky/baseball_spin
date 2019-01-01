#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <math.h>

#include "edge_detect.h"
#include "load_data.h"
#include "detect_baseball.h"

using namespace cv;
using namespace std;


void makeRot(Mat& R) {
  
   //cout << R.col(1).t()*R.col(1) << "\n";
   Mat col0 = R.col(0);
   Mat col1 = R.col(1);
   Mat col2 = R.col(2);

   for (int i = 0; i < 3; i++) {
      Mat curr = R.col(i);
      for (int j = 0; j<i; j++) {
         Mat prev = R.col(j);
         double scale = prev.dot(curr);
         curr = curr-scale*prev;
      }
      double norm = sqrt(curr.dot(curr));
      curr = curr/norm;
      R.col(i) = curr;
   }
   //cout << R.t()*R << "\n";
}


void rescale_edge(
   const Mat& edge,
   Mat& out,
   float r,
   float cx,
   float cy) {

   float new_r = 20.0;
   float new_cx = 24;
   float new_cy = 24;

   /*Modify the edge to standardize it and speed up optimization*/
   out = Mat::zeros(Size(49,49),edge.type());
   Mat nonZeroCoordinates;
   
   findNonZero(edge, nonZeroCoordinates);
   Point coord;
   for (int i = 0; i < nonZeroCoordinates.total(); i++ ) {
      coord = nonZeroCoordinates.at<Point>(i);
      int newx = round((coord.x-cx)*(new_r/r)+new_cx); 
      int newy = round((coord.y-cy)*(new_r/r)+new_cy); 
      out.at<char>(newy,newx) = 255;
   }
}

// This follows the algorithm described in the slides
void estimate_orientation(
   const Mat& edge, // Raw edge input
   Mat& xyz, // Estimated 3D points of the seam 
   float r, // radius of ball
   // origin of the ball
   float cx, 
   float cy,
   float min_dist, // Minimum distance between projected seam and detected edge
   const Mat& xyz_rotated,
   const Mat& cost_matrix,
   Mat& newEdge) {
   
   /*
   This takes in the raw edge fragments (which hopefully includes the seam 
   fragments) and uses the 3D model of the seam to eliminate fragments that 
   cannot possibly be part of the seam. This means that blemishes that are not
   eliminated and happen to fall in the right area can confuse our algorithm.
   We need to test these scenarios...

   The output of this is a cleaned up binary image of the seams AND the 
   3D orientation of the model.
   */

   /* Given a 3D model of a seam, we first project it to a 2D image. We call this
   the projected seam.

   For every pixel in the computed edge fragments (obtained in get_seam_pix),
   we compute its distance to the closest point in the projected seam. If 
   distance is smaller than min_dist, we consider that edge fragment pixel valid.
   We repeat this for every pixel in the edge fragment and total up the number of 
   valid pixels.

   We then do this for every orientation of the 3D model and take the 3D 
   orientation that yields the highest number of valid pixels.
   
   The computation here is weird looking because I have pre-computed 
   the orientations (xyz_rotated), and the the computation of the minimum distance
   from the edge fragments to the projected seam (cost_matrix)  
   */
   Mat edge_rescaled;
   rescale_edge(edge,edge_rescaled,r,cx,cy);
   edge_rescaled.convertTo(edge_rescaled,CV_32F);
   int n = cost_matrix.rows;
   // Flatten the edge into one-channel, one-row and repeat
   // for n rows
   Mat tmp = Mat::ones(Size(1,n),edge_rescaled.type())*
      edge_rescaled.reshape(1,1);
   // Compute the best cost
   Mat prod_tmp = cost_matrix.mul(tmp) < 255*min_dist;
   Mat prod;
   prod_tmp.convertTo(prod,CV_32F);
   Mat res; 
   reduce(prod,res,1,CV_REDUCE_SUM);
   
   // Take the 3D orientation that has the highest number of valid pixels
   double min,max;
   Point min_loc,max_loc;
   minMaxLoc(res, &min, &max, &min_loc, &max_loc);
   
   int indx = max_loc.y;
   xyz = xyz_rotated(Rect(0,3*indx,396,3));
    
   /*
   In addition to the orientation of the 3D model, I know remove 
   all the computed edge fragments that do not intersect the projected
     seam pixels. 
   */
   newEdge = Mat::zeros(edge.size(),edge.type());
   Mat labels;
   Mat stats;
   Mat centroids;
   int numComponents = connectedComponentsWithStats(edge,labels,stats,
      centroids,4);
   int numPts = 396;//#xyz_curr.cols;
   int largest_area = 0;
   
   for ( int i = 0; i < numPts; i++ ) {
      float z = xyz.at<float>(2,i);
      if (z > 0) {
         int x = r*xyz.at<float>(0,i)+cx;
         int y = r*xyz.at<float>(1,i)+cy;
         int label_tmp = labels.at<int>(y,x);   
         if (label_tmp != 0)
            newEdge += labels==label_tmp;
      }
   }
   newEdge = newEdge > 0;
}


float estimate_rotation(
   const Mat& xyz, // Initial orientation 
   Mat& bestR, // Best radius
   const vector<Mat>& edge_vec, // All seam pixels
   const vector<float>& r_vec, // All baseball radius
   const vector<Point>& cent_vec, // All centers
   int start,
   const Mat& Rmat2,
   int numIter // Number of rotations to try
   ) {

   float minCost = 1e100; // Arbitraily large

   int numRot = numIter;//1000;
   int numEdges = edge_vec.size();
 
   /* For every frame, we get the x and y coordinates for every 
     pixel in the computed seams. We will use this to compute the 
   distance to the projected seam   
   */
   vector<Mat> x_nzCoord;
   vector<Mat> y_nzCoord;
   for (int ii = 0; ii < numEdges; ii++) {
      //Mat nztmp;
      vector<float> xx_vec;
      vector<float> yy_vec;
      Mat tmp_edge = edge_vec[ii];
      int total = 0;

      for (int y = 0; y < tmp_edge.rows; y++) {
         for (int x = 0; x < tmp_edge.cols; x++) {
            if (tmp_edge.at<char>(y,x)) {
               xx_vec.push_back(x);
               yy_vec.push_back(y);
               total++;
            }
         }
      }
      Mat xx_Mat(total,1,CV_32F);
      Mat yy_Mat(total,1,CV_32F);
      
      memcpy(xx_Mat.data, xx_vec.data(), total*sizeof(float));
      memcpy(yy_Mat.data, yy_vec.data(), total*sizeof(float));
      
      x_nzCoord.push_back(xx_Mat);
      y_nzCoord.push_back(yy_Mat);
   }
   
   
   Mat xyz_tmp;
   
   for (int jj = 0; jj < numRot; jj += 1) {
      Mat Rtmp = Rmat2(Rect(0,3*jj,3,3));
      xyz.copyTo(xyz_tmp);
      
      float tmpCost = 0;
      // Hard-coded....should change this
      for (int ii = start+1; ii < start+4; ii++) {
         // First we rotate the 3D model using the current rotation
         xyz_tmp = Rtmp*xyz_tmp;
         float rtmp = r_vec[ii];
         float cx = cent_vec[ii].x;
         float cy = cent_vec[ii].y;
         vector<float> x_data;
         vector<float> y_data;
         int numNz = 0;
        
         // We then get the x and y coordinates of all the projected seam 
         //pixels

         // This hack speeds things up 
         for (int kk = 0; kk < 396; kk += 10) {
            if (xyz_tmp.at<float>(2,kk) > 0) {
               x_data.push_back( rtmp*xyz_tmp.at<float>(0,kk)+cx );
               y_data.push_back( rtmp*xyz_tmp.at<float>(1,kk)+cy );
               numNz++;
            }
         }
         Mat xxdata(1,numNz,CV_32F); 
         Mat yydata(1,numNz,CV_32F);
         memcpy(xxdata.data, x_data.data(), numNz*sizeof(float));
         memcpy(yydata.data, y_data.data(), numNz*sizeof(float));
        
         Mat xx = x_nzCoord[ii];
         Mat yy = y_nzCoord[ii];
         
         // We only consider consective frames that all have *some* 
         // computed seam fragments
         // This is the linear algebra where for each pixel of the computed
         // seam, we compute the distance to the closest projected seam
         // and add it up for every computed seam pixel
         int scale = xx.rows;
         if (scale != 0) {
            int num_nz = numNz;
            Mat dx = Mat::ones(Size(1,scale),CV_32F)*xxdata-xx*Mat::ones(Size(num_nz,1),CV_32F);
            Mat dx2 = dx.mul(dx);
            Mat dy = Mat::ones(Size(1,scale),CV_32F)*yydata-yy*Mat::ones(Size(num_nz,1),CV_32F);
            Mat dy2 = dy.mul(dy);
            Mat err_tmp = dx2+dy2;
            Mat err;
            sqrt(err_tmp,err);
            //err = err.pow(0.5);
            Mat err2;
            reduce(err,err2,1,CV_REDUCE_MIN);
        
            // Get the max value
            Mat maxMat;
            reduce(err2,maxMat,0,CV_REDUCE_MAX);
            if (maxMat.at<float>(0) > 5e2) { 
               tmpCost = 1e10;
               break;
            }
         
            Mat sumMat;
            reduce(err2,sumMat,0,CV_REDUCE_SUM);
            tmpCost += (1.0*sumMat.at<float>(0)/numNz);
            }  else {
               tmpCost += 1e100;
            }
      }

      if (tmpCost < minCost) {
         minCost = tmpCost;
         Rtmp.copyTo(bestR);
      }
   }

   return minCost; // We return both the smallest cost and the bestR
}

float getSpin(const Mat& R, int fps) {
   float trR = trace(R)[0];
   float angle = acos(0.5*(trR-1));
   float PI = 3.1415926;
   return angle/(2*PI)*fps*60;
}

void getAxis(const Mat& R, vector<float>& axis) {
   float h = R.at<float>(2,1);
   float f = R.at<float>(1,2);
   float c = R.at<float>(0,2);
   float g = R.at<float>(2,0);
   float d = R.at<float>(1,0);
   float b = R.at<float>(0,1);
   axis[0] = h-f;
   axis[1] = c-g;
   axis[2] = d-b;
   float norm = sqrt(pow(h-f,2)+pow(c-g,2)+pow(d-b,2));
   axis[0] = axis[0]/norm;
   axis[1] = axis[1]/norm;
   axis[2] = axis[2]/norm;
}

int visualize = 0;

int main(int argc, char **argv) {
 
   char * path_to_frames = argv[1];
   int start_indx = stoi(argv[2]);
   //char * prefix = "";//argv[2];  
   //if (prefix  == " ") prefix = "";
   //cout << path_to_frames << "\n";
   //cout << prefix << "\n";
   
   // Parameters for seam detection
   // Size of the bilateral/laplacian filter used to detect 
   // the edge. Ranges between 3 and 7 result in reasonable output
   // See edge_detect in edge_detect.cpp 
   int filter_size = stoi(argv[3]); 
   // Threshold for the logos. In the videos we have seen, logos are dark
   // This corresponds to low pixel intensities. This does the following
   // in python-esque code img[img<logo_thresh] = 0
   // Our algorithm ignores pixels taht are set to 0
   // See get_seam_pix in edge_detect.cpp
   int logo_thresh = stoi(argv[4]); 
   // Minimium fragment size of potential seam
   // See get_seam_pix in edge_detect.cpp 
   int min_size = stoi(argv[5]); // Minimimum size of fragment
   // This is used to align the 3D model of the edge to the edge fragment.
   // We say that an edge fragment pixel matches the projected seam
   // if the distance from the fragment pixel to the closest projected seam
   // is within 1 pixel  
   float min_dist = stof(argv[6]); 
   // Threshold to convert the Laplacian output into 
   // a binary image where each pixel corresponds to a potential seam
   float lap_thresh = stof(argv[7]);
   
   visualize = stoi(argv[8]); // Visualize the estimated seams
   int numR2try = stoi(argv[9]); // Number of rotations to try


   // Load the resource files
   // These matrices contain pre-computed 3D seam orientations,
   // all of the rotation matrices, etc.
   Mat Rmat;
   convert_to_mat("rotations.bin",Rmat);
   Mat xyz;
   convert_to_mat("seams.bin",xyz);
   Mat cost_matrix;
   convert_to_mat("cost_matrix.bin",cost_matrix);
   Mat xyz_rotated;
   convert_to_mat("xyz_rotated.bin",xyz_rotated);
   Mat Rmat2;
   convert_to_mat("rotations2.bin",Rmat2);
   
   // Pre-process and load in the data as specified by the user
   vector<Mat> im_vec;
   vector<Mat> edge_vec;
   vector<Mat> xyz_vec;
   vector<float> r_vec;
   vector<Point> cent_vec;
   
   process_data(path_to_frames,start_indx,im_vec,r_vec,cent_vec);
   
   cout << "Video " << argv[1] << "\n";

   // For each frame, detect the seams and align the 3D model
   for (int ii = 0; ii < im_vec.size(); ii++) {
      float r = r_vec[ii];
      float cx = cent_vec[ii].x;
      float cy = cent_vec[ii].y;
      Mat im_tmp;
      if (r==0) im_tmp = Mat::zeros(24,24,CV_8UC1);
      else im_tmp = im_vec[ii]; 
      Mat edge;
     
      // Get both the correctly oriented 3D model and the edge 
      get_seam_pix(im_tmp,r,cx,cy,filter_size,logo_thresh,min_size,lap_thresh,edge);
      Mat xyz;
      Mat new_edge;
      estimate_orientation(edge,xyz,r,cx,cy,min_dist,xyz_rotated,cost_matrix,new_edge);
      edge_vec.push_back(new_edge);
      xyz_vec.push_back(xyz);
      
      
      if (visualize) {
         cout << "Frame " << ii << "\n";
         imshow("img",im_tmp);
         imshow("raw seam", edge);
         imshow("filtered seam",new_edge);
         waitKey(0); 
      }
   }
  
   // Estimate the spin
  /*
     Basic idea is we take 4 consecutive frames, where for each frame we 
     have an oriented 3D model and the pixels that we believe to be the seam. The 
     goal is to find a rotation matrix, R, so that when we apply it sequentially
     to the first orientated model, its projection "matches" the seam pixels for
     the subsequent frames. 

     The process is as follows. We first apply the rotation to the initial 3D 
     orientation. We then project it to the same image as the seam in the next frame.     We can measure how well this projection  matches the seams in the next frame 
     by taking every pixel of the seam, and computing 
     its distance to the closest point on the projection. We sum up the distance 
     for every pixel and divide by the total number of pixels. 

     We then apply the rotation again and repeat this process for the next frame..
    */ 
   
   /*
      We repeat the above process over a sliding window, and take the rotation
      that has the smallest error.
   */
   float bestErr = 1e100;
   float bestSpin = 0;
   Mat bestR;
   for (int start = 0; start < xyz_vec.size()-3; start++) {
      Mat R;
      Mat xyz_tmp;
      xyz_vec[start].copyTo(xyz_tmp);
      float minCost = estimate_rotation(xyz_tmp,R,edge_vec,r_vec,cent_vec,start,Rmat2,numR2try);
     int numEdge = 0;
      for (int kk = start; kk < start+4; kk++) {
         //cout << edge_vec[kk] << "\n";
              numEdge += sum(edge_vec[kk])[0]/255;
     }
      
      cout << minCost << "," << numEdge << "," << getSpin(R,240) << "\n";  
      if ( (numEdge > 200) && (minCost < bestErr)) {
         bestErr = minCost;
         R.copyTo(bestR);
         bestSpin = getSpin(R,240);
      }
      
   }
   
   if (bestSpin == 0) {
      cout << "Image quality is too low. Estimation failed!\n\n";
      return -1;
   }
   // Convert rotation into spin and spin axis
   vector<float> axis(3);
   getAxis(bestR,axis);
   cout << "Best error is: " << bestErr << "\n";
   cout << "Spin is: " << bestSpin << "\n"; 
   cout << "Axis is: [" << axis[0] << "," << axis[1] << "," << axis[2] << "]\n\n";
}

