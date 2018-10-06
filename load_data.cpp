#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

#include "load_data.h"

using namespace cv;
using namespace std;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void read_data(
   const char* filename,
   vector<float>& out) {
   string line;
   ifstream fileObj(filename);
   if ( fileObj.is_open() ) {
      while ( getline(fileObj,line) ) {
         out.push_back( stof(line) );
      }
      fileObj.close();
   }
}

void convert_to_mat(
   const char* filename,
   Mat& out) {
   
   // Read binary
   ifstream in(filename,ios_base::binary);
   unsigned vsize;
   in.read(reinterpret_cast<char *>(&vsize),sizeof(unsigned));
   vector<float> raster(vsize);
   in.read(reinterpret_cast<char *>(&raster[0]),vsize*sizeof(float));
   in.close();
   
   int h = raster[0];
   int w = raster[1];
   out = Mat(h,w,CV_32F);
   int indx = 2;
   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         out.at<float>(y,x) = (float)raster[indx];
         indx++;
      }
   }
}

/*
int main() {

   //Mat mat;
   //convert_to_mat("cost_matrix.txt",mat);
  
   vector<float> raster;
   read_data("seams.txt",raster);

   ofstream o("seams.bin",ios_base::binary);
   unsigned Xsize=raster.size();
   o.write (reinterpret_cast<char *>(&Xsize),sizeof(unsigned));
   o.write(reinterpret_cast<char *>(&raster[0]), raster.size()*sizeof(float));
   o.close();
} */
