#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>
#include <opencv2/face.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv::face;
using namespace cv::xfeatures2d;

void run() {
    String file_location = "Media/";
    vector<String> fn;
    glob(file_location + "Ground Truth/np", fn, false);
    vector<Mat> images;
    size_t count = fn.size();
    Mat skinSamples = imread(file_location + "SkinSamples.jpg");

    cout << count + "\n";
    for (size_t i = 0; i < count; i++) {
        images.push_back(imread(fn[i]));
    }


    vector<CascadeClassifier> cascades;
    String cascade_files[] = { "haarcascades/haarcascade_frontalface_alt2.xml",
                                "haarcascades/haarcascade_eye.xml",
                                "haarcascades/haarcascade_mcs_mouth.xml",
                                "haarcascades/custom_mask_classifier.xml"};
    int number_of_cascades = sizeof(cascade_files) / sizeof(cascade_files[0]);
    for (int cascade_file_no = 0; (cascade_file_no < number_of_cascades); cascade_file_no++)
    {
        CascadeClassifier cascade;
        string filename(file_location);
        filename.append(cascade_files[cascade_file_no]);
        
        if (!cascade.load(filename))
        {
            cout << "Cannot load cascade file: " << filename << endl;
            return;
        }
        
        cascades.push_back(cascade);
    }
    
    for (int i = 0; i < images.size(); i++) {
        detectMaskedFaces(images[i], cascades[0]/*, skinSamples */);
    }  
}

Mat detectMaskedFaces(Mat image, CascadeClassifier cascades/*, Mat skinSamples*/) {
    vector<Rect> faces;
    int width, height, x, y;
    Mat haarImage = haarFaceDetection(image, cascades, faces);
    if (faces.size() != 0) {
        width = faces[0].width; height = faces[0].height; x = faces[0].x; y = faces[0].y;
        Rect topHalfFace(x, y, width, height / 2);
        Rect bottomHalfFace(x, y + (height / 2), width, height / 2);

        Mat gray, backProjectedFace, colourThresholdSkin;
        //backProjectedFace = backProject(skinSamples, haarImage);
        //threshold(backProjectedFace, backProjectedFace, 5, 255, THRESH_BINARY);
        colourThresholdSkin = detectSkin(haarImage);
        vector<double> skinProbablities = countPixels(colourThresholdSkin, topHalfFace, bottomHalfFace);
        String s;
        if (skinProbablities[0] > 50 && skinProbablities[1] < 50) {
            s = "Masked";
        }
        else if (skinProbablities[0] > 50 && skinProbablities[1] > 50) {
            s = "Unmasked";
        }
        else {
            s = "Unknown";
        }
        rectangle(image, faces[0], Scalar(0, 0, 255), 2);
        putText(image, s, Point((faces[0].x + 20), (faces[0].y + 20)), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 4);
        vector<Mat> vec = {/* backProjectedFace ,*/ haarImage , image, colourThresholdSkin};
        return image;
    }
}

vector<double> countPixels(Mat skinPixels, Rect topHalfFace, Rect bottomHalfFace) {
    Mat topFace = skinPixels(topHalfFace);
    Mat bottomFace = skinPixels(bottomHalfFace);
    double skinPixelsTop = ((double) countNonZero(topFace) / (double) topFace.total()) * 100.0;
    double skinPixelsBottom = ((double) countNonZero(bottomFace) / (double) bottomFace.total()) * 100.0;
    cout << "top: " + to_string(skinPixelsTop) + "%\n" ;
    cout << "bottom: " + to_string(skinPixelsBottom) + "%\n";
    vector<double> skinProbablities = { skinPixelsTop , skinPixelsBottom };
    return skinProbablities;
}

Mat detectSkin(Mat input) {
    //attempt at skin detection using YCrCb colour space
    Mat image_YCrCb;
    cvtColor(input, image_YCrCb, COLOR_BGR2YCrCb);
    Mat mask, skinRegion;
    inRange(image_YCrCb, Scalar(0,133,77), Scalar(255,173,127), mask);

    input.copyTo(skinRegion, mask);
    return mask;
}

Mat haarFaceDetection(Mat image, CascadeClassifier cascade, vector<Rect> &faces) {
    if (faces.empty()) {
        faces.clear();
    }

    Mat gray, interestArea;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    cascade.detectMultiScale(gray, faces, 1.05, 3, cv::CASCADE_FIND_BIGGEST_OBJECT, Size(30, 30));
    Mat mask = image.clone();
    mask.setTo(cv::Scalar(0, 0, 0));
    cout << faces.size() + "\n";
    if (faces.size() != 0) {
        Rect mostProminentFace = faces[0];
        if (faces.size() > 1) {
            for (int i = 1; i < faces.size(); i++) {
                if ((faces[i].width + faces[i].height) > (faces[i - 1].width + faces[i - 1].height)) {
                    mostProminentFace = faces[i];
                }
            }
        }
        for (int count = 0; count < (int)faces.size(); count++) {
            rectangle(mask, mostProminentFace, cv::Scalar(255, 255, 255), cv::FILLED, 8, 0);
        }
        faces[0] = mostProminentFace;
    }
    
    image.copyTo(interestArea, mask);
    return interestArea;
}

/*
void detectFacemarks(vector<Mat> images, vector<CascadeClassifier> cascades) {
    const string facemark_filename = "lbfmodel.yaml";
    Ptr<Facemark> facemark = createFacemarkLBF();
    facemark->loadModel(facemark_filename);
    cout << "Loaded facemark LBF model" << endl;
    vector<Rect> faces;

    for (int i = 0; i < images.size(); i++) {
        faceDetector(images[i], faces, cascades[0]);
        if (faces.size() != 0) {
            cv::rectangle(images[0], faces[0], Scalar(255, 0, 0), 2);
            vector<vector<Point2f> > shapes;
            if (facemark->fit(images[i], faces, shapes)) {
                drawFacemarks(images[i], shapes[0], cv::Scalar(0, 0, 255));
                vector<Mat> vec = { images[i] };
                Mat out = makeCanvas(vec, 600, 1);
                imshow("", out);
                char c = waitKey();
            }
        }
        else {
            vector<Mat> vec = { images[i] };
            Mat out = makeCanvas(vec, 600, 1);
            imshow("", out);
            char c = waitKey();
            cout << "Faces not detected. " + i << endl;
        }
    }
}

void faceDetector(const Mat& image, vector<Rect>& faces, CascadeClassifier& face_cascade) {
    Mat gray;
    if (image.channels() > 1) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }
    equalizeHist(gray, gray);
    faces.clear();
    face_cascade.detectMultiScale(gray, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, Size(30, 30));
}

void featureMatching(Mat trainImage) {
    //surf implementation
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints;
    detector->detect(trainImage, keypoints);
    //-- Draw keypoints
    Mat img_keypoints;
    drawKeypoints(trainImage, keypoints, img_keypoints);
    imshow("SURF Keypoints", img_keypoints);
    char c = waitKey();
}
*/