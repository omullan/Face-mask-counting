#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
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
using namespace cv::ml;

Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);
Mat* gaussianMixture(VideoCapture video);
void haarFaceDetection(Mat image, vector<Rect>& faces);
void runMedianBackground(VideoCapture video, float learningRate, int valuesPerBin);
void run();
void faceDetector(const Mat image, vector<Rect>& faces, Net net);
Mat detectFacemarks(Mat image, Net net, Ptr<Facemark> facemark);
Mat detectSkin(Mat input);
void featureMatching(Mat trainImage);
void floodFillPostprocess(Mat& img, const Scalar& colorDiff);
Mat videoFaceDetection(Mat image, CascadeClassifier cascade);
Mat backProject(Mat samples, Mat input);
vector<double> countPixels(Mat skinPixels, Rect topHalfFace, Rect bottomHalfFace);
Mat detectMaskedFaces(Mat image, Net net, vector<String>& result,  Ptr<Boost> boost, vector<Rect>& faces);
void writeVideoToFile(Mat* frames, String fileName, int fps, int width, int height, int noOfFrames);
Mat eyeDetector(Mat image, CascadeClassifier face_cascade);
Net load();
void DNNfaceDetect(Net net, Mat image, vector<Rect> &faces, float confidenceThreshold);
double faceHistogram(Mat input, Rect topHalfFace, Rect bottomHalfFace, vector<Mat>& vec);
Ptr<Facemark> loadFacemarkModel();
void gradientImage(Mat input, Rect topHalfFace, Rect bottomHalfFace);
template <typename _Tp> void ELBP(const Mat& src, Mat& dst, int radius, int neighbors);
void train_svm_hog_descriptor();
void test_svm_classifier(vector<Mat> images, int testSize);
bool boosted_mask_classifier(Ptr<Boost> boost, Mat input);
void extract();
Mat* runWithoutBM(VideoCapture video);
vector<vector<int>> readCSV(String filename);
map<int, vector<int>> readMAFAGT();
Mat drawGroundTruth(Mat image, map<int, vector<int>> groundtruth, int key);
vector<int> compareResults(map<int, vector<int>> groundtruth, int key, vector<Rect> faces, vector<String> result, Mat image);
double getHistogram(Mat input1, Mat input2);