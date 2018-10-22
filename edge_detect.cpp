#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "edge_detect.h"

using namespace cv;
using namespace std;

void edge_detect(
   const Mat& im, // baseball img
   int d, // Bilateral Filter params
   double sigmaC,
   double sigmaS,
   Mat& edge
   ) {
   /* Get the edges that correspond the baseball seam and others.
    * Used as a first pass, and output needs to be cleaned
    */
   Mat smooth;
   bilateralFilter(im,smooth,d,sigmaC,sigmaS);
   Laplacian(smooth,edge,CV_64F,5); // Assume kernel size = 5 
}

void get_seam_pix(
   const Mat& im, // gray-scale image of croppxed baseball
   float r, // radius of the baseball
   float cx, // pixel location, x, of the baseball
   float cy, // pixel location, y, of the baseball
   Mat& seam_pix // seam pixels
   ) {
   /*
    * Get the pixels that correspond to a baseball's seams by performing
    * edge detection and morphological filtering
    */

   // Some useful parameters
   int minSize = 20; // Minimum size for the  seam
   float fracSize = 0.4; // Size for the second seam

   int h = im.rows;
   int w = im.cols;
   
   // Remove the potential junk around the baseball
   Mat mask = Mat::zeros(Size(w,h),im.type());
   circle(mask,Point(cx,cy),0.90*r,Scalar(255,255,255),-1,8,0);

   Mat im_clean;
   im.copyTo(im_clean,mask);
   // Detect the seam and other edges in the image
   Mat edge;
   edge_detect(im_clean,5,75,75,edge);
   
   // Threshold to get a binary img of the seam
   edge = edge > 0; 
   Mat edge_nb;
   edge.copyTo(edge_nb,mask);
   
   // Identify the connected components...one of which is the seam
   // we want to get
   Mat labels;
   Mat stats;
   Mat centroids;
   int numComponents = connectedComponentsWithStats(edge_nb,labels,stats,
      centroids,8);

   // Get second and third largest index
   // Clearly not the most elegant way to do this
   int first = 0;
   int second = 0;
   int third = 0;
   int ii1,ii2,ii3;
   int tmp;
   for (int ii = 0; ii < numComponents; ii++) {
      tmp = stats.at<int>(ii,CC_STAT_AREA);
      if (tmp > third) {
         if (tmp > second) {
            if (tmp > first) {
               third = second;
               ii3 = ii2;
               second = first;
               ii2 = ii1;
               first = tmp;
               ii1 = ii;
            } else {
               third = second;
               ii3 = ii2;
               second = tmp;
               ii2 = ii;
            }
         } else {
            third = tmp;
            ii3 = ii;
         }
      }
   }
   //cout << "======\n";
   //cout << first << "\n";
   //cout << second << "\n";
   //cout << third << "\n";
   //cout << labels.size() << "\n"; 

   if (second > minSize) {
      seam_pix = labels == ii2;
      if ( (numComponents > 2) && (third > fracSize*second) ) {
         seam_pix += labels == ii3;
      }
   } else {
      seam_pix = Mat::zeros(Size(w,h),im.type());
   }
   //seam_pix = labels == ii2;
}


/*
int main() {
   float r = 21.1032581329;
   float cx = 42;
   float cy = 42;

   vector<float> out;
   read_data("seams.txt",out);
   
   Mat im = imread("im5.png",0);
   cout << type2str(im.type()) << "\n";
   //Mat out;
   //get_seam_pix(im,r,cx,cy,out);
   Mat seam_proj = Mat::zeros(im.size(),im.type());
   int numPts = out.size()/3;
   cout << numPts<< "\n";
   for (int i = 0; i < numPts; i++) {
      float x = out[3*i];
      float y = out[3*i+1];
      float z = out[3*i+2];
      if (z > 0) {
         seam_proj.at<char>(round(r*y+cy),round(r*x+cx)) = 255;
      }
   }
   //imshow("proj",seam_proj);
   //waitKey(0);
}*/
