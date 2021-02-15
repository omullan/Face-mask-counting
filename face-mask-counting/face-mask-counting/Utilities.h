#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <iostream>

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);
Mat* gaussianMixture(VideoCapture video, CascadeClassifier cascade);
Mat haarFaceDetection(Mat image, CascadeClassifier cascade, vector<Rect>& faces);
void runMedianBackground(VideoCapture video, float learningRate, int valuesPerBin, CascadeClassifier cascade);
void run();
void faceDetector(const Mat& image, vector<Rect>& faces, CascadeClassifier& face_cascade);
void detectFacemarks(vector<Mat> images, vector<CascadeClassifier> cascades);
Mat detectSkin(Mat input);
void featureMatching(Mat trainImage);
void floodFillPostprocess(Mat& img, const Scalar& colorDiff);
Mat videoFaceDetection(Mat image, CascadeClassifier cascade);
Mat backProject(Mat samples, Mat input);