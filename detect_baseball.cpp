#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void detect_ball(
   const Mat& frame1,
   const Mat& frame2,
   Mat& ball_frame,
   float& r,
   Point& cent) {

   
   Mat diff = frame2-frame1;
   int numBallPix = sum(diff > 100)[0];
   if (numBallPix > 150) {
      // We have a ball or something here
      Mat bg = diff < 30;
      Mat strel = getStructuringElement(MORPH_RECT,Size(5,5));
      Mat bg_erode; 
      erode(bg,bg_erode,strel);
      Mat frame2_tmp;
      frame2.copyTo(frame2_tmp,bg_erode==0);
       
      // Let's get the ball only
      Mat nonbg = bg_erode==0;
      Mat labels;
      Mat stats;
      Mat centroids;
      int numComponents = connectedComponentsWithStats(nonbg,labels,stats,
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
      // Get the bounding box 
      int x = stats.at<int>(ii2,CC_STAT_LEFT);
      int y = stats.at<int>(ii2,CC_STAT_TOP);
      int w = stats.at<int>(ii2,CC_STAT_WIDTH);
      int h = stats.at<int>(ii2,CC_STAT_HEIGHT);
      ball_frame = frame2_tmp(Rect(x,y,w,h));
      
      // Get approximate radius and center
      r = 0.25*(w+h);
      cent = Point(r,r);
   }
}

void process_data(
   int vidIndx,
   vector<Mat>& im_vec,
   vector<float>& r_vec,
   vector<Point>& cent_vec) {
   
   string root = string("frames/video");
   root += to_string(vidIndx);
   root += "_120fps_frame";
   string ext = string(".png");

   try {
      int i = 0;
      Mat im_prev;
      Mat im = imread(root+to_string(i)+ext,0);
      i = 1;
      while (true) {
         im.copyTo(im_prev);
         im = imread(root+to_string(i)+ext,0);
         if (im.cols == 0) break;
         Mat ball_frame;
         float r;
         Point cent;
         detect_ball(im_prev,im,ball_frame,r,cent);
         im_vec.push_back(ball_frame);
         r_vec.push_back(r);
         cent_vec.push_back(cent);
         i++;
      }
   } catch (...) {}
}

/*
int main() {
   vector<Mat> im_vec;
   vector<float> r_vec;
   vector<Point> cent_vec;
   process_data(3,im_vec,r_vec,cent_vec);
}*/
