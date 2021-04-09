#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>

const int TP = 0;
const int TN = 1;
const int FP = 2;
const int FN = 3;
const int INV = 4;

void run() {
    String file_location = "Media/";
    vector<String> fn;
    //glob(file_location + "Ground Truth/Datasets/MAFA/test-images/images", fn, false);
    //glob(file_location + "Ground Truth/singlefacetesting", fn, false);
    glob(file_location + "Ground Truth/Datasets/lfw/1", fn, false);
    vector<Mat> images;
    size_t count = fn.size();

    cout << count + "\n";
    for (size_t i = 0; i < 1000; i++) {
        images.push_back(imread(fn[i]));
        cout << i << "\n";
    }
    Ptr<Boost> boost = Boost::create();
    boost = StatModel::load<Boost>("ADABOOST_TEST_4.xml");
    if (boost->empty()) {
        cout << "could not load SVM";
        return;
    }
    Net net = load();
    int maskedCorrectCount = 0;
    int unmaskedCorrectCount = 0;
    int maskedFaceNotFoundCount = 0;
    int unmaskedFaceNotFoundCount = 0;
    map<int, vector<int>> groundtruth = readMAFAGT();
    int gtMasked = 0;
    int gtUnmasked = 0;
    int gtInvalid = 0;
    for (int i = 1; i < groundtruth.size() + 1 ; i++) {
        vector<int> points = groundtruth.at(i);
        int faceCount = points.size() / 18;
        for (int j = 0; j < faceCount; j++) {
            int offset = j * 18;
            if (points[4 + offset] == 3 || points[13 + offset] == 1 || points[13 + offset] == 5) {
                gtInvalid++;
            }
            else if (points[4 + offset] == 1) {
                gtMasked++;
            }
            else if (points[4 + offset] == 2) {
                gtUnmasked++;
            }
        }
    }
    cout << gtMasked << " " << gtUnmasked << " " << gtInvalid << "\n";
    if(images.size() == 200) {
        vector<Mat> detectedFaces;
        for (int i = 0; i < images.size(); i++) {
            cout << i << "\n";
            vector<String> result;
            vector<Rect> faces;
            Mat out = detectMaskedFaces(images[i], net, result, boost, faces);
            detectedFaces.push_back(out);
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
        
        vector<Mat> vec = { detectedFaces[11], detectedFaces[27], detectedFaces[37],detectedFaces[46],detectedFaces[105],detectedFaces[116],detectedFaces[128],detectedFaces[170] };
        Mat out = makeCanvas(vec, 600, 2);
        imshow("", out);
        waitKey();
        imwrite("Media/Ground Truth/classificationresults7.png", out);
        
        cout << "\n" << "True postive rate masked: " << maskedCorrectCount;
        cout << "\n" << "False negative rate masked: " << (100 - maskedCorrectCount);
        cout << "\n" << "Masked faces not found: " << maskedFaceNotFoundCount;
        cout << "\n" << "True Negative rate unmasked: " << unmaskedCorrectCount;
        cout << "\n" << "False positive rate unmasked: " << (100 - unmaskedCorrectCount);
        cout << "\n" << "Unasked faces not found: " << unmaskedFaceNotFoundCount << "\n";
    }
    else if (count == 13233) {

    }
    else {
        vector<int> evalMetrics = { 0,0,0,0,0 };
        
        for (int i = 1; i < images.size(); i++) {
            cout << i << "\n";
            vector<String> result;
            vector<Rect> faces;
            Mat algo_image = detectMaskedFaces(images[i], net, result, boost, faces);
            Mat gt_image = drawGroundTruth(images[i], groundtruth, i+1);
            vector<Mat> vec = { algo_image, gt_image };
            Mat out = makeCanvas(vec, 600, 2);
            imshow("", out);
            waitKey();
            
            vector<int> comparisonResults = compareResults(groundtruth, i + 1, faces, result, images[i]);
            for (int i = 0; i < comparisonResults.size(); i++) {
                evalMetrics[i] += comparisonResults[i];
            }
        } 
        cout << "\n";
        cout << "tp: " << evalMetrics[TP] << "\n";
        cout << "tn: " << evalMetrics[TN] << "\n";
        cout << "fp: " << evalMetrics[FP] << "\n";
        cout << "fn: " << evalMetrics[FN] << "\n";
        cout << "actual tp: " << gtMasked << "\n";
        cout << "actual tn: " << gtUnmasked << "\n";
        cout << "faces missed: " << ((gtMasked + gtUnmasked) - (evalMetrics[TP] + evalMetrics[TN] + evalMetrics[FP] + evalMetrics[FN])) << "\n";
        cout << "invalid / false faces: " << evalMetrics[INV] << "\n";
    }
}

vector<int> compareResults(map<int, vector<int>> groundtruth, int key, vector<Rect> faces, vector<String> result, Mat image) {
    vector<int> points = groundtruth.at(key);
    int faceCount = points.size() / 18;
    vector<int> evalMetric = { 0,0,0,0,0 };
    Mat cloneImage = image.clone();
    for (int i = 0; i < faceCount; i++) {
        int offset = i * 18;
        Rect gt_face(points[0 + offset], points[1 + offset], points[2 + offset], points[3 + offset]);
        String gt_result;
        bool hasMatch = false;
        if (points[4 + offset] == 3 || points[13 + offset] == 1 || points[13 + offset] == 5 || points[10 + offset] == 4) {
            gt_result = "Invalid";
        }
        else if (points[4 + offset] == 1) {
            gt_result = "Masked";
        }
        else if (points[4 + offset] == 2) {
            gt_result = "Unmasked";
        }
        for (int j = 0; j < faces.size(); j++) {
            if (!(image.size().width <= (gt_face.x + gt_face.width)) && !(image.size().height <= (gt_face.y + gt_face.height)) && (gt_face.x > 0) && (gt_face.y > 0)) {
                Rect intersection = (faces[j] & gt_face);
                int faceArea = max(faces[j].area(), gt_face.area());
                double overlapArea = ((double)intersection.area() / (double)faceArea);
                if (overlapArea > 0.4) {
                    if (gt_result == "Masked" && result[j] == "Masked") {
                        evalMetric[TP]++;
                        rectangle(cloneImage, faces[j], Scalar(31, 255, 0), 2);
                        putText(cloneImage, result[j], Point((faces[j].x), (faces[j].y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(31, 255, 0), 1.5);
                        hasMatch = true;
                    }
                    else if (gt_result == "Unmasked" && result[j] == "Unmasked") {
                        evalMetric[TN]++;
                        rectangle(cloneImage, faces[j], Scalar(31, 255, 0), 2);
                        putText(cloneImage, result[j], Point((faces[j].x), (faces[j].y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(31, 255, 0), 1.5);
                        hasMatch = true;
                    }
                    else if (gt_result == "Unmasked" && result[j] == "Masked") {
                        evalMetric[FP]++;
                        rectangle(cloneImage, faces[j], Scalar(0, 0, 255), 2);
                        putText(cloneImage, result[j], Point((faces[j].x), (faces[j].y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255), 1.5);
                        hasMatch = true;
                    }
                    else if (gt_result == "Masked" && result[j] == "Unmasked") {
                        evalMetric[FN]++;
                        rectangle(cloneImage, faces[j], Scalar(0, 0, 255), 2);
                        putText(cloneImage, result[j], Point((faces[j].x), (faces[j].y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255), 1.5);
                        hasMatch = true;
                    }
                    
                }
            }
        }
        if (hasMatch == false) {
            if (gt_result != "Invalid") {
                rectangle(cloneImage, gt_face, Scalar(0, 0, 255), 2);
                putText(cloneImage, "Missed", Point((gt_face.x), (gt_face.y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255), 1.5);
            }
        }
    }
    for (int i = 0; i < faces.size(); i++) {
        bool matchFound = false;
        for (int j = 0; j < faceCount; j++) {
            int offset = j * 18;
            Rect gt_face(points[0 + offset], points[1 + offset], points[2 + offset], points[3 + offset]);
            String gt_result;
            if (points[4 + offset] == 3 || points[13 + offset] == 1 || points[13 + offset] == 5 || points[10 + offset] == 4) {
                gt_result = "Invalid";
            }
            else if (points[4 + offset] == 1) {
                gt_result = "Masked";
            }
            else if (points[4 + offset] == 2) {
                gt_result = "Unmasked";
            }
            if (!(image.size().width <= (gt_face.x + gt_face.width)) && !(image.size().height <= (gt_face.y + gt_face.height)) && (gt_face.x > 0) && (gt_face.y > 0)) {
                Rect intersection = (faces[i] & gt_face);
                int faceArea = max(faces[i].area(), gt_face.area());
                double overlapArea = ((double)intersection.area() / (double)faceArea);
                if (overlapArea > 0.4) {
                    matchFound = true;
                } 
            }
        }
        if (matchFound == false) {
            evalMetric[INV]++;
            /*
            Mat cloneImage = image.clone();
            for (int k = 0; k < faces.size(); k++) {
                if (result[k] == "Masked") {
                    rectangle(cloneImage, faces[k], Scalar(0, 0, 255), 2);
                    putText(cloneImage, result[k], Point((faces[k].x), (faces[k].y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255), 1.5);
                }
                else {
                    rectangle(cloneImage, faces[k], Scalar(255, 0, 255), 2);
                    putText(cloneImage, result[k], Point((faces[k].x), (faces[k].y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 0, 255), 1.5);
                }
            }
            Mat gt_image = drawGroundTruth(image, groundtruth, key);
            vector<Mat> vec = { cloneImage, image(faces[i]) , gt_image };
            Mat out = makeCanvas(vec, 600, 2);
            imwrite("Media/Ground Truth/invalidfaces/invalid_face_image_" + to_string(key) + "_face_" + to_string(i) + ".png", out);
            */
        }
    }
    vector<Mat> vec = { cloneImage };
    Mat out = makeCanvas(vec, 600, 2);
    imshow("", out);
    waitKey();
    
    return evalMetric;
}

Mat drawGroundTruth(Mat image, map<int, vector<int>> groundtruth, int key) {
    vector<int> points = groundtruth.at(key);
    Mat tmp = image.clone();
    int faceCount = points.size() / 18;
    for (int i = 0; i < faceCount; i++) {
        int offset = i * 18;
        Rect face(points[0 + offset], points[1 + offset], points[2 + offset], points[3 + offset]);
        if (points[4 + offset] == 1) {
            rectangle(tmp, face, Scalar(0, 0, 255), 2);
            putText(tmp, "Masked", Point((points[0 + offset]), (points[1 + offset] + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255), 1.5);
        }
        else if (points[4 + offset] == 2){
            rectangle(tmp, face, Scalar(255, 0, 255), 2);
            putText(tmp, "Unmasked", Point((points[0 + offset]), (points[1 + offset] + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 0, 255), 1.5);
        }
        else {
            rectangle(tmp, face, Scalar(255, 0, 0), 2);
            putText(tmp, "invalid", Point((points[0 + offset]), (points[1 + offset] + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255), 1.5);
        }
        
    }
    return tmp;
}

Mat detectMaskedFaces(Mat image, Net net, vector<String> &result, Ptr<Boost> boost, vector<Rect> &faces) {
    faces.clear();
    int width, height, x, y;
    float confidenceThreshold = 0.5;
    Mat cloneImage = image.clone();
    DNNfaceDetect(net, image, faces, confidenceThreshold);
    //haarFaceDetection(image, faces);
    if (faces.size() != 0) {
        for (int i = 0; i < faces.size(); i++) {
            Mat haarImage = cloneImage(faces[i]);
            width = faces[i].width; height = faces[i].height; x = faces[i].x; y = faces[i].y;
            Rect topHalfFace(x, y, width, height / 2);
            Rect bottomHalfFace(x, y + (height / 2), width, height / 2);
            
            vector<Mat> hists;
            double histMatchingScore = faceHistogram(image, topHalfFace, bottomHalfFace, hists);

            Mat colourThresholdSkin = detectSkin(cloneImage);
            vector<double> skinProbablities = countPixels(colourThresholdSkin, topHalfFace, bottomHalfFace);
            bool is_mask = boosted_mask_classifier(boost, haarImage);
            
            double unmaskScore = 0.0;
            if (!is_mask) {
                unmaskScore += 0.3;
            }
            if (histMatchingScore > 0.0 && histMatchingScore < 0.5) {
                //cout << "hist match: " << histMatchingScore;
                if (histMatchingScore <= 0.2) {
                    unmaskScore += 0.3;
                }
                else {
                    unmaskScore += (0.3 - (histMatchingScore - 0.2));
                }
            }
            if (skinProbablities[0] > 50 && skinProbablities[1] > 50) {
                float tmp = ((skinProbablities[0] + skinProbablities[1]) - 100) / 333.33;
                //cout << skinProbablities[0] << " " << skinProbablities[0] << "\n";
                //cout << tmp << "\n";
                unmaskScore += (0.1 + tmp);
            }
            //cout << "Unmask score: " << unmaskScore << "\n";
            if (unmaskScore >= 0.5) {
                result.push_back("Unmasked");
            }
            else {
                result.push_back("Masked");
            }
            int int_unmaskScore = (int)(unmaskScore * 100);

            if (result[i] == "Masked") {
                //rectangle(cloneImage, faces[i], Scalar(0,0,255), 2);
                rectangle(cloneImage, faces[i], Scalar(31, 255, 0), 2);
                putText(cloneImage, result[i], Point((faces[i].x + 5), (faces[i].y + 30)), FONT_HERSHEY_DUPLEX, 1.5, Scalar(31, 255, 0),2);
                putText(cloneImage, to_string((100 - (int_unmaskScore))) + "%", Point((faces[i].x), (faces[i].y - 10)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(31, 255, 0), 1.5);
            }
            else {
                //rectangle(cloneImage, faces[i], Scalar(0, 0, 255), 2);
                
                rectangle(cloneImage, faces[i], Scalar(31, 255, 0), 2);
                putText(cloneImage, result[i], Point((faces[i].x + 5), (faces[i].y + 20)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(31, 255, 0),1.5);
                putText(cloneImage, to_string(100 -(int_unmaskScore)) + "%", Point((faces[i].x), (faces[i].y - 10)), FONT_HERSHEY_DUPLEX, 0.7, Scalar(31, 255, 0), 1.5);
                
            }
            /*
            vector<Mat> vec = { hists[0], hists[1], cloneImage, hists[4] , hists[5] };
            Mat out = makeCanvas(vec, 600, 2);
            imshow("", out);
            waitKey();
            cout << histMatchingScore << "\n";
            */
            /*
            vector<Mat> vec = {cloneImage , colourThresholdSkin(topHalfFace),colourThresholdSkin, colourThresholdSkin(bottomHalfFace) };
            Mat out = makeCanvas(vec, 600, 2);
            imshow("", out);
            waitKey();
           
            
            vector<Mat> vec = {image,cloneImage };
            Mat out = makeCanvas(vec, 600, 1);
            imshow("", out);
            waitKey();
             */
        }
        return cloneImage;
    }
    else {
        /*
        imshow("", image);
        char c = waitKey();
        */
        return cloneImage;
    }
}

vector<double> countPixels(Mat skinPixels, Rect topHalfFace, Rect bottomHalfFace) {
    Mat topFace = skinPixels(topHalfFace);
    Mat bottomFace = skinPixels(bottomHalfFace);
    double skinPixelsTop = ((double) countNonZero(topFace) / (double) topFace.total()) * 100.0;
    double skinPixelsBottom = ((double) countNonZero(bottomFace) / (double) bottomFace.total()) * 100.0;
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

void haarFaceDetection(Mat image, vector<Rect> &faces) {
    CascadeClassifier cascade;
    cascade.load("Media/haarcascades/haarcascade_frontalface_alt2.xml");
    if (!faces.empty()) {
        faces.clear();
    }
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    cascade.detectMultiScale(gray, faces, 1.4, 3, cv::CASCADE_FIND_BIGGEST_OBJECT, Size(30, 30));
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
