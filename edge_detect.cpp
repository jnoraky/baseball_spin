#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "edge_detect.h"

using namespace cv;
using namespace std;

void edge_detect(
   const Mat& im, // Cropped image of the baseball
   int d, // Filter size 
   double sigmaC, // Bilateral filter params
   double sigmaS, // Bilateral filter params
   Mat& edge
   ) {
   /* This does the initial edge detection of the ball image.
      This is a first pass, where we get edges that correspond 
      to the seam, blemishes, logos, and the ball edge.
      The output is the laplacian which is another image that positive 
      for a transition from light to dark aka regions around the seam
   */
  
   Mat smooth;
   //im.copyTo(smooth);
   bilateralFilter(im,smooth,d,75,75);
   //GaussianBlur(im,smooth,Size(d,d),0,0);
   //medianBlur(im,smooth,d);
   Laplacian(smooth,edge,CV_64F,d); // Assume kernel size = 5 
}


void get_seam_pix(
   const Mat& im, // Cropped Image of baseball
   float r, // Radius of the baseball
   float cx, // Center x coord
   float cy, // Center y coord
   int filter_size, // Size of bilateral and laplacian filter
   int logo_thresh, // Logos are darker, and 
   // we only consider pixels brighter than this threshold 
   int min_size, // Minimum fragment size to be considered "seam"
   float lap_thresh, // Laplacian threshold to binarize edge
   Mat& seam_pix             
   ){

   /*
      This is the first function used to roughly detect the seam pixels.
      We first use  a crude edge detection to find all the candidate
      seam pixels. We then ignore small fragments, fragments that correspond
     to dark pixels (logos).  
   */   
   int h = im.rows;
   int w = im.cols;
  
   if (r == 0) {
      seam_pix = Mat::zeros(h,w,im.type());
      return;
   }
   
   // Remove potential junk around the baseball image
   // by keeping only the pixels within 80% of the baseball radius
   // and setting all the other pixels to 0, which our algorithm
   // ignores
   Mat mask = Mat::zeros(Size(w,h),im.type());
   circle(mask,Point(cx,cy),0.80*r,Scalar(255,255,255),-1,8,0);
   
   // Perform the crude edge detection see comments above
   Mat edge;
   edge_detect(im,filter_size,75,75,edge);
   // We threshold the laplacian to get a binary img of the seam
   // At this point we have a binary image where non-zero values
   // correspond to potential seam fragments
   Mat edge_tmp = edge > lap_thresh; 
   Mat edge_nb;
   edge_tmp.copyTo(edge_nb,mask);
   
   // Because edge detection is not perfect, all of the seams
   // May not be connected. So our image is binary image composed of 
   // different connected components that may be seams or not.
   // We use OpenCV to identify them and compute interesting statistics
   // Like the area
   Mat labels;
   Mat stats;
   Mat centroids;
   int numComponents = connectedComponentsWithStats(edge_nb,labels,stats,
      centroids,4);
   
   // This is just an idiosyncracy of the opencv function. We do the next set of 
   //lines to retain all the fragments above a certain size (min_size pixels)

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
   
   Mat seam_pix_tmp = Mat::zeros(Size(w,h),im.type());
   for (int ii = 0; ii < numComponents; ii++) {
      tmp = stats.at<int>(ii,CC_STAT_AREA);
      if ( (tmp > min_size) && (tmp <= second) ) {
         seam_pix_tmp += labels == ii;
      }
   } 

   // Finally we ignore all pixels that correspond that correspond 
   // to dark patches
   mask = im > logo_thresh;
   seam_pix_tmp.copyTo(seam_pix,mask);
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
