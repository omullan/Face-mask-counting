#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <iostream>
#include "opencv2/dnn.hpp"
#include <opencv2/face.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv::face;
using namespace std;
using namespace cv;
using namespace cv::dnn;

Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);
Mat* gaussianMixture(VideoCapture video, vector<CascadeClassifier> cascade);
Mat haarFaceDetection(Mat image, CascadeClassifier cascade, vector<Rect>& faces);
void runMedianBackground(VideoCapture video, float learningRate, int valuesPerBin, CascadeClassifier cascade);
void run();
void faceDetector(const Mat image, vector<Rect>& faces, Net net);
Mat detectFacemarks(Mat image, Net net, Ptr<Facemark> facemark);
Mat detectSkin(Mat input);
void featureMatching(Mat trainImage);
void floodFillPostprocess(Mat& img, const Scalar& colorDiff);
Mat videoFaceDetection(Mat image, CascadeClassifier cascade);
Mat backProject(Mat samples, Mat input);
vector<double> countPixels(Mat skinPixels, Rect topHalfFace, Rect bottomHalfFace);
Mat detectMaskedFaces(Mat image, vector<CascadeClassifier> cascades, Mat skinSamples,
    Net net, Ptr<Facemark> facemark, String& result);
void writeVideoToFile(Mat* frames, String fileName, int fps, int width, int height, int noOfFrames);
Mat eyeDetector(Mat image, CascadeClassifier face_cascade);
Net load();
Mat DNNfaceDetect(Net net, Mat image, vector<Rect> &faces);
double faceHistogram(Mat input, Rect topHalfFace, Rect bottomHalfFace);
Ptr<Facemark> loadFacemarkModel();