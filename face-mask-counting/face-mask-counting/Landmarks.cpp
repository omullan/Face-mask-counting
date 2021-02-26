#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>

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
                                "haarcascades/haarcascade_eye_tree_eyeglasses.xml",
                                "haarcascades/haarcascade_mcs_mouth.xml"};
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
    Net net = load();
    Ptr<Facemark> facemark = loadFacemarkModel();
    int maskedCorrectCount = 0;
    int unmaskedCorrectCount = 0;
    for (int i = 0; i < images.size(); i++) {
        String result;
        Mat out = detectMaskedFaces(images[i], cascades, skinSamples, net, facemark, result);
        if (i < 100) {
            if (result == "Masked") {
                maskedCorrectCount++;
            }
        }
        else {
            if (result == "Unmasked") {
                unmaskedCorrectCount++;
            }
        }
    }  
    cout << "\n" << "True postive rate masked: " << maskedCorrectCount;
    cout << "\n" << "False negative rate masked: " << (100 - maskedCorrectCount);
    cout << "\n" << "True postive rate unmasked: " << unmaskedCorrectCount;
    cout << "\n" << "False negative rate unmasked: " << (100 - unmaskedCorrectCount);

}

Mat detectMaskedFaces(Mat image, vector<CascadeClassifier> cascades, Mat skinSamples, 
    Net net, Ptr<Facemark> facemark, String &result) {
    vector<Rect> faces;
    int width, height, x, y;
    Mat haarImage = DNNfaceDetect(net, image, faces);
    if (faces.size() != 0) {
        cout << faces.size();
        width = faces[0].width; height = faces[0].height; x = faces[0].x; y = faces[0].y;
        Rect topHalfFace(x, y, width, height / 2);
        Rect bottomHalfFace(x, y + (height / 2), width, height / 2);
        double histMatchingScore = faceHistogram(image, topHalfFace, bottomHalfFace);

        Mat gray, backProjectedFace, colourThresholdSkin, facemarkImage;
        facemarkImage = detectFacemarks(image, net, facemark);
        backProjectedFace = backProject(skinSamples, haarImage);
        threshold(backProjectedFace, backProjectedFace, 5, 255, THRESH_BINARY);
        colourThresholdSkin = detectSkin(haarImage);
        vector<double> skinProbablities = countPixels(colourThresholdSkin, topHalfFace, bottomHalfFace);
        /*
        if (skinProbablities[0] > 40 && skinProbablities[1] < 50) {
            result = "Masked";
        }
        else if (skinProbablities[0] > 40 && skinProbablities[1] > 50) {
            result = "Unmasked";
        }
        else {
            result = "Unknown";
        }
        */
        if (histMatchingScore > 0.4) {
            result = "Masked";
        }
        else {
            result = "Unmasked";
        }

        rectangle(image, faces[0], Scalar(0, 0, 255), 2);
        putText(image, result, Point((faces[0].x + 20), (faces[0].y + 20)), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 4);
        /*
        vector<Mat> vec = {haarImage , image, colourThresholdSkin, facemarkImage };
        Mat out = makeCanvas(vec, 600, 2);
        imshow("", out);
        char c = waitKey();
        */
        return image;
    }
    else {
        /*
        imshow("", image);
        char c = waitKey();
        */
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

Ptr<Facemark> loadFacemarkModel() {
    const string facemark_filename = "lbfmodel.yaml";
    Ptr<Facemark> facemark = createFacemarkLBF();
    facemark->loadModel(facemark_filename);
    return facemark;
}

Mat detectFacemarks(Mat image, Net net, Ptr<Facemark> facemark) {
    vector<Rect> faces;
    faceDetector(image, faces, net);
    if (faces.size() != 0) {
        cv::rectangle(image, faces[0], Scalar(255, 0, 0), 2);
        vector<vector<Point2f> > shapes;
        if (facemark->fit(image, faces, shapes)) {
            drawFacemarks(image, shapes[0], cv::Scalar(0, 0, 255));
            /*
            vector<Mat> vec = { image };
            Mat out = makeCanvas(vec, 600, 1);
            imshow("", out);
            char c = waitKey();
            */
        }
    }
    return image;
}

void faceDetector(const Mat image, vector<Rect>& faces, Net net) {
    DNNfaceDetect(net, image, faces);
}

