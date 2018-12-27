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
   
   // Find the lowest 
   double min,max;
   Point min_loc,max_loc;
   minMaxLoc(res, &min, &max, &min_loc, &max_loc);
   
   int indx = max_loc.y;
   xyz = xyz_rotated(Rect(0,3*indx,396,3));
    
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


void estimate_orientation(
   const Mat& edge,
   Mat& xyz,
   float r,
   float cx,
   float cy,
   const Mat& xyz_rotated,
   const Mat& cost_matrix) {

   Mat edge_rescaled;
   rescale_edge(edge,edge_rescaled,r,cx,cy);
   edge_rescaled.convertTo(edge_rescaled,CV_32F);


   int n = cost_matrix.rows;
   // Flatten the edge into one-channel, one-row and repeat
   // for n rows
   Mat tmp = Mat::ones(Size(1,n),edge_rescaled.type())*
      edge_rescaled.reshape(1,1);
   
   // Compute the best cost
   Mat prod = cost_matrix.mul(tmp);
   Mat res; 
   reduce(prod,res,1,CV_REDUCE_SUM);
   
   // Find the lowest 
   double min,max;
   Point min_loc,max_loc;
   minMaxLoc(res, &min, &max, &min_loc, &max_loc);
   
   int indx = min_loc.y;
   xyz = xyz_rotated(Rect(0,3*indx,396,3));
  
   /*
   Mat xyz_curr = xyz; 
   // For debugging purposes
   r = 20;
   cx = 24;
   cy = 24;
   Mat est_edge = Mat::zeros(edge_rescaled.size(),edge_rescaled.type());
   edge_rescaled.copyTo(est_edge);
   //cout << xyz_curr.size() << "\n";
   int numPts = 396;//#xyz_curr.cols;
   for ( int i = 0; i < numPts; i++ ) {
      float z = xyz_curr.at<float>(2,i);
      if (z > 0) {
         int x = r*xyz_curr.at<float>(0,i)+cx;
         int y = r*xyz_curr.at<float>(1,i)+cy;
         est_edge.at<float>(y,x) = 255;
      }
   }
   imshow("edge", edge_rescaled);
   imshow("est", est_edge);
   waitKey(0);*/
}

float estimate_rotation(
   const Mat& xyz,
   Mat& bestR,
   const vector<Mat>& edge_vec,
   const vector<float>& r_vec,
   const vector<Point>& cent_vec,
   int start,
   const Mat& Rmat2,
   int numIter,
   float min_dist
   ) {

   float minCost = 0;//1e50; // Arbitraily large

   int numRot = numIter;//1000;
   int numEdges = edge_vec.size();
 
   // Get a vector of the non-zero x and y coordinates
   // of the edges
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
   
   //Mat Rmat2;
   //convert_to_mat("rotations2.bin",Rmat2);
   // Get vector of non-zero
   Mat xyz_tmp;
   
   for (int jj = 0; jj < numRot; jj += 1) {
      Mat Rtmp = Rmat2(Rect(0,3*jj,3,3));
      xyz.copyTo(xyz_tmp);
      
      float tmpCost = 0;
      // Hard-coded....should change this
      for (int ii = start+1; ii < start+4; ii++) {
         xyz_tmp = Rtmp*xyz_tmp;
         float rtmp = r_vec[ii];
         float cx = cent_vec[ii].x;
         float cy = cent_vec[ii].y;
         
         // Prepare the non-zero coordinates of the current 3D seam
         vector<float> x_data;
         vector<float> y_data;
         int numNz = 0;
         
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
         int scale = xx.rows;
         int num_nz = numNz;
         Mat dx = Mat::ones(Size(1,scale),CV_32F)*xxdata-xx*Mat::ones(Size(num_nz,1),CV_32F);
         Mat dx2 = dx.mul(dx);
         Mat dy = Mat::ones(Size(1,scale),CV_32F)*yydata-yy*Mat::ones(Size(num_nz,1),CV_32F);
         Mat dy2 = dy.mul(dy);
         Mat err = dx2+dy2;
         Mat err2;
         reduce(err,err2,1,CV_REDUCE_MIN);
         /* 
         // Get the max value
         Mat maxMat;
         reduce(err2,maxMat,0,CV_REDUCE_MAX);
         if (maxMat.at<float>(0) > 5e2) { 
            //cout << "here!!!\n";
            tmpCost = 1e10;
            break;
         }*/
         //cout << err2 << "\n";
         Mat err2_tmp = err2 < min_dist;
         Mat err2_tmp2;
         err2_tmp.convertTo(err2_tmp2,CV_32F);
         Mat sumMat;
         reduce(err2_tmp2,sumMat,0,CV_REDUCE_SUM);
         //cout << sumMat << "\n";
         tmpCost += (1.0*sumMat.at<float>(0));
      }

      if (tmpCost > minCost) {
         minCost = tmpCost;
         Rtmp.copyTo(bestR);
      }
   }
   return minCost;
}


float estimate_rotation(
   const Mat& xyz,
   Mat& bestR,
   const vector<Mat>& edge_vec,
   const vector<float>& r_vec,
   const vector<Point>& cent_vec,
   int start,
   const Mat& Rmat2,
   int numIter
   ) {

   float minCost = 1e100; // Arbitraily large

   int numRot = numIter;//1000;
   int numEdges = edge_vec.size();
 
   // Get a vector of the non-zero x and y coordinates
   // of the edges
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
   //cout << "here\n"; 
   
   //Mat Rmat2;
   //convert_to_mat("rotations2.bin",Rmat2);
   // Get vector of non-zero
   Mat xyz_tmp;
   
   for (int jj = 0; jj < numRot; jj += 1) {
      Mat Rtmp = Rmat2(Rect(0,3*jj,3,3));
      xyz.copyTo(xyz_tmp);
      
      float tmpCost = 0;
      // Hard-coded....should change this
      for (int ii = start+1; ii < start+4; ii++) {
         xyz_tmp = Rtmp*xyz_tmp;
         float rtmp = r_vec[ii];
         float cx = cent_vec[ii].x;
         float cy = cent_vec[ii].y;
         //cout << rtmp << "\n";   
         // Prepare the non-zero coordinates of the current 3D seam
         vector<float> x_data;
         vector<float> y_data;
         int numNz = 0;
         
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
         
         int scale = xx.rows;
         //cout << scale << "\n";
         if (scale != 0) {
         int num_nz = numNz;
         Mat dx = Mat::ones(Size(1,scale),CV_32F)*xxdata-xx*Mat::ones(Size(num_nz,1),CV_32F);
         Mat dx2 = dx.mul(dx);
         Mat dy = Mat::ones(Size(1,scale),CV_32F)*yydata-yy*Mat::ones(Size(num_nz,1),CV_32F);
         Mat dy2 = dy.mul(dy);
         Mat err = dx2+dy2;
         Mat err2;
         reduce(err,err2,1,CV_REDUCE_MIN);
        
         
         // Get the max value
         Mat maxMat;
         reduce(err2,maxMat,0,CV_REDUCE_MAX);
         if (maxMat.at<float>(0) > 5e2) { 
            //cout << "here!!!\n";
            tmpCost = 1e10;
            break;
         }
         //cout << "boom\n";
         Mat sumMat;
         reduce(err2,sumMat,0,CV_REDUCE_SUM);
         tmpCost += (1.0*sumMat.at<float>(0)/numNz);
         } else {
            tmpCost += 1e100;
         }
      }

      if (tmpCost < minCost) {
         minCost = tmpCost;
         Rtmp.copyTo(bestR);
      }
   }
   return minCost;
}

float getSpin(const Mat& R, int fps) {
   //float trR = R.at<float>(0,0)+R.at<float>(1,1)+R.at<float>(2,2);
   //cout << R.t()*R << "\n";
   //makeRot(R);
   //cout << R.t()*R << "\n";
 
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
}

int visualize = 0;
int main(int argc, char **argv) {
   
   // Parameters for seam detection
   int filter_size = 5; // Size of the bilateral/laplacian filter used to detect edge
   int logo_thresh = 0; // Threshold for dark pixels that are typically logos. Ignore pixels below this intensity
   int min_size = 20; // Minimimum size of fragment
   float min_dist = 1; // Parameter used to estimate the seam. see PPT
   float lap_thresh = 0; // Laplacian threshold to binarize output of edge detection
   visualize = 1; // Visualize the estimated seams

   // Load the matrices and resource files
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
   
   // Pre-process and load in the data
   vector<Mat> im_vec;
   vector<Mat> edge_vec;
   vector<Mat> xyz_vec;
   vector<float> r_vec;
   vector<Point> cent_vec;
   
   process_data(stoi(argv[1]),im_vec,r_vec,cent_vec);
   /*reverse(im_vec.begin(),im_vec.end());
   reverse(r_vec.begin(),r_vec.end());
   reverse(cent_vec.begin(),cent_vec.end());
   */
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
      
      get_seam_pix(im_tmp,r,cx,cy,filter_size,logo_thresh,min_size,lap_thresh,edge);
      //cout << "here\n"; 
      Mat xyz;
      Mat new_edge;
      estimate_orientation(edge,xyz,r,cx,cy,min_dist,xyz_rotated,cost_matrix,new_edge);
      
      edge_vec.push_back(new_edge);
      xyz_vec.push_back(xyz);
      
     // Uncomment to see results of the seam detection
      
      if (visualize) {
         cout << "Frame " << ii << "\n";
         imshow("img",im_tmp);
         imshow("raw seam", edge);
         imshow("filtered seam",new_edge);
         waitKey(0); 
      }
   }
   //cout << "Done\n"; 
   // Estimate the spin here
   float bestErr = 1e100;
   //float bestErr = 0;
   float bestSpin = 0;
   Mat bestR;
   for (int start = 0; start < xyz_vec.size()-3; start++) {
      //cout << start << "\n";
      Mat R;
      Mat xyz_tmp;
      xyz_vec[start].copyTo(xyz_tmp);
      
      float minCost = estimate_rotation(xyz_tmp,R,edge_vec,r_vec,cent_vec,start,Rmat2,stoi(argv[2]));
      
      if (minCost < bestErr) {
         bestErr = minCost;
         R.copyTo(bestR);
         bestSpin = getSpin(R,240);
      }
      /*
      float maxCost = estimate_rotation(xyz_tmp,R,edge_vec,r_vec,cent_vec,start,Rmat2,stoi(argv[2]),3);
      if (maxCost > bestErr) {
         bestErr = maxCost;
         R.copyTo(bestR);
         bestSpin = getSpin(R,120);
      }*/
      
   }
   // Output everything
   vector<float> axis(3);
   getAxis(bestR,axis);
   cout << "Best error is: " << bestErr << "\n";
   cout << "Spin is: " << bestSpin << "\n"; 
   cout << "Axis is: [" << axis[0] << "," << axis[1] << "," << axis[2] << "]\n";
}

