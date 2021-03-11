#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>

void run() {
    String file_location = "Media/";
    vector<String> fn;
    glob(file_location + "Ground Truth/Datasets/MAFA/test-images/images", fn, false);
    vector<Mat> images;
    size_t count = fn.size();

    cout << count + "\n";
    for (size_t i = 0; i < count; i++) {
        images.push_back(imread(fn[i]));
        cout << i << "\n";
    }
    Ptr<Boost> boost = Boost::create();
    boost = StatModel::load<Boost>("ADABOOST_TEST_3.xml");
    if (boost->empty()) {
        cout << "could not load SVM";
        return;
    }
    Net net = load();
    int maskedCorrectCount = 0;
    int unmaskedCorrectCount = 0;
    int maskedFaceNotFoundCount = 0;
    int unmaskedFaceNotFoundCount = 0;
    for (int i = 0; i < images.size(); i++) {
        vector<String> result;
        Mat out = detectMaskedFaces(images[i], net, result, boost);
        if(result.size() != 0) {
            if (i < 100) {
                if (result[0] == "Masked") {
                    maskedCorrectCount++;
                }
            }
            else {
                if (result[0] == "Unmasked") {
                    unmaskedCorrectCount++;
                }
            }
        }
        else {
            if (i < 100) {
                maskedFaceNotFoundCount++;
            }
            else {
                unmaskedFaceNotFoundCount++;
            }
        }
    }  
    cout << "\n" << "True postive rate masked: " << maskedCorrectCount;
    cout << "\n" << "False negative rate masked: " << (100 - maskedCorrectCount);
    cout << "\n" << "Masked faces not found: " << maskedFaceNotFoundCount;
    cout << "\n" << "True postive rate unmasked: " << unmaskedCorrectCount;
    cout << "\n" << "False negative rate unmasked: " << (100 - unmaskedCorrectCount);
    cout << "\n" << "Unasked faces not found: " << unmaskedFaceNotFoundCount << "\n";


}

Mat detectMaskedFaces(Mat image, Net net, vector<String> &result, Ptr<Boost> boost) {
    vector<Rect> faces;
    int width, height, x, y;
    float confidenceThreshold = 0.5;
    DNNfaceDetect(net, image, faces, confidenceThreshold);
    if (faces.size() != 0) {
        cout << faces.size() << "\n";
        for (int i = 0; i < faces.size(); i++) {
            Mat haarImage = image(faces[i]);
            cout << faces.size();
            width = faces[i].width; height = faces[i].height; x = faces[i].x; y = faces[i].y;
            Rect topHalfFace(x, y, width, height / 2);
            Rect bottomHalfFace(x, y + (height / 2), width, height / 2);
            //double histMatchingScore = faceHistogram(image, topHalfFace, bottomHalfFace);
            //gradientImage(image, topHalfFace, bottomHalfFace);
            //LBPFace(image, topHalfFace, bottomHalfFace);
            bool is_mask = boosted_mask_classifier(boost, haarImage);
            //Mat gray, colourThresholdSkin;
            //colourThresholdSkin = detectSkin(haarImage);
            //vector<double> skinProbablities = countPixels(colourThresholdSkin, topHalfFace, bottomHalfFace);
            if (is_mask) {
                result.push_back("Masked");
            }
            else {
                result.push_back("Unmasked");
            }

            /*
            if (skinProbablities[0] > 40 && skinProbablities[1] < 50) {
                result = "Masked";
            
            }
            else if (skinProbablities[0] > 40 && skinProbablities[1] > 50) {
                result = "Unmasked";
            }
            if (histMatchingScore > 0.4) {
                result = "Masked";
                cout << result;
            }
            else {
                result = "Unmasked";
                cout << result;
            }
            */
            rectangle(image, faces[i], Scalar(0, 0, 255), 2);
            putText(image, result[i], Point((faces[i].x + 20), (faces[i].y + 20)), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 4);
            
            vector<Mat> vec = {haarImage , image};
            Mat out = makeCanvas(vec, 600, 2);
            imshow("", out);
            char c = waitKey();
    
        }
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
    float confidenceThreshold = 0.5;
    DNNfaceDetect(net, image, faces, confidenceThreshold);
}

bool boosted_mask_classifier(Ptr<Boost> boost, Mat input) {
    HOGDescriptor hog(Size(50, 50), Size(10, 10), Size(5, 5), Size(10, 10),
        9, 1, -1, HOGDescriptor::L2Hys, 0.2,
        false, HOGDescriptor::DEFAULT_NLEVELS, false);
    Mat image;
    resize(input, image, Size(100, 100));
    vector<float> descriptors;
    hog.compute(image, descriptors, Size(8, 8));
    Mat testDescriptor = Mat::zeros(1, descriptors.size(), CV_32F);
    for (int j = 0; j < descriptors.size(); j++) {
        testDescriptor.at<float>(0, j) = descriptors[j];
    }
    float label = boost->predict(testDescriptor);
    //float label = svm->predict(testDescriptor);
    //cout << label << "\n";
    
    bool slabel = false;
    if (label > 0) {
        slabel = true;
    }
    return slabel;
    
}
