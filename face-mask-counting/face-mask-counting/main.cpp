#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>


int main(int argc, char* argv[])
{
    String file_location = "Media/";
    VideoCapture capture(file_location + "backgarden2.avi");
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open" << endl;
        return 0;
    }
   
    int frameCount = capture.get(CAP_PROP_FRAME_COUNT);
    

    //extract();
    //train_svm_hog_descriptor();
    run();
    //Mat* frames = gaussianMixture(capture);
    /*
    Mat* frames = runWithoutBM(capture);
    
    writeVideoToFile(frames, "output9.avi", 30, 527, 600, frameCount-5);
    for (int i = 0; i < (capture.get(CAP_PROP_FRAME_COUNT) - 5); i++) {
        imshow("", frames[i]);
        char c = waitKey(10);
    }
    */
   
    //runMedianBackground(capture, (float)1.005, 1, cascades[0]);
}
Mat* gaussianMixture(VideoCapture video) {
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
    Net net = load();
    Mat element(2, 2, CV_8U, Scalar(1));
    
    Mat frame, mask;
    int frameCount = video.get(CAP_PROP_FRAME_COUNT);
    cout << frameCount;
    Mat* masks = new Mat[frameCount - 5];
    Ptr<Boost> boost = Boost::create();
    boost = StatModel::load<Boost>("ADABOOST_TEST_2.xml");
    if (boost->empty()) {
        cout << "could not load SVM";
        return masks;
    }
    video >> frame;
    int faceDetectCounter = 0;
    Mat faceDetect = frame.clone();
    bool firstFaceDetected = false;
    for (int i = 0; i < frameCount - 5 && !frame.empty(); i++) {
        pBackSub->apply(frame, mask);

        rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
            cv::Scalar(255, 255, 255), -1);
        stringstream ss;
        ss << video.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
            FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        Mat cleanedImage, thresholdedImage, temp, foreground;
        threshold(mask, thresholdedImage, 150, 255, THRESH_BINARY);
        morphologyEx(thresholdedImage, temp, MORPH_CLOSE, element);
        morphologyEx(temp, cleanedImage, MORPH_OPEN, element);
        frame.copyTo(foreground, cleanedImage);

        if (faceDetectCounter == 5 || firstFaceDetected) {
            vector<String> result;
            Mat tmp = detectMaskedFaces(foreground, net, result, boost);
            if (!tmp.empty()) {
                faceDetect = tmp;
            }   
            faceDetectCounter = 0;
        }

        vector<Mat> vec = { frame, cleanedImage, foreground, faceDetect };
        
        Mat out = makeCanvas(vec, 600, 2);
        
        imshow("", out);
        char c = waitKey(1);
        cout << out.size();
        
        masks[i] = out;
        video >> frame;
        faceDetectCounter++;
        cout << "gmm: " << i << "\n";
    }
    return masks;
}

Mat* runWithoutBM(VideoCapture video) {
    Net net = load();

    Mat frame, mask;
    int frameCount = video.get(CAP_PROP_FRAME_COUNT);
    cout << frameCount;
    Mat* masks = new Mat[frameCount - 5];
    Ptr<Boost> boost = Boost::create();
    boost = StatModel::load<Boost>("ADABOOST_TEST_3.xml");
    if (boost->empty()) {
        cout << "could not load model";
        return masks;
    }
    video >> frame;
    int faceDetectCounter = 0;
    Mat faceDetect = frame.clone();
    bool firstFaceDetected = false;
    for (int i = 0; i < frameCount - 5 && !frame.empty(); i++) {
        rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
            cv::Scalar(255, 255, 255), -1);
        stringstream ss;
        ss << video.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
            FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        vector<Rect> faces;
        float confidenceThreshold = 0.5;
        vector<String> result;
        /*
        DNNfaceDetect(net, frame, faces, confidenceThreshold);
        if (faces.size() != 0) {
            faceDetect = detectMaskedFaces(frame, net, result, boost);
        }
        */
        faceDetect = detectMaskedFaces(frame, net, result, boost);
        vector<Mat> vec = { frame, faceDetect };
        Mat out = makeCanvas(vec, 600, 2);
        imshow("", out);
        char c = waitKey(1);
        masks[i] = out;
        cout << out.size();
        video >> frame;
        cout << "frame: " << i << "\n";
    }
    return masks;
}

void writeVideoToFile(Mat* frames, String fileName, int fps, int width, int height, int noOfFrames) {
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cout << "h: " << height << " w: " << width;
    VideoWriter output(fileName, codec, fps, Size(width, height), true);
    if (!output.isOpened()) {
        cout << "Could not open the output video file for write\n";
        return;
    }
    cout << "opened";
    for (int i = 0; i < noOfFrames; i++) {
        output.write(frames[i]);
        cout << i << "\n";
    }
}

void test_svm_classifier(vector<Mat> images, int testSize) {
    HOGDescriptor hog(Size(50, 50), Size(10, 10), Size(5, 5), Size(10, 10),
        9, 1, -1, HOGDescriptor::L2Hys, 0.2,
        false, HOGDescriptor::DEFAULT_NLEVELS, false);
    Ptr<SVM> svm = SVM::load("HOG_TEST_1.xml");
    if (svm->empty()) {
        cout << "could not load SVM";
        return;
    }
    for (int i = 0; i < testSize; i++) {
        Mat image;
        resize(images[i], image, Size(100, 100));
        vector<float> descriptors;
        hog.compute(image, descriptors, Size(8, 8));
        Mat testDescriptor = Mat::zeros(1, descriptors.size(), CV_32F);
        for (int j = 0; j < descriptors.size(); j++) {
            testDescriptor.at<float>(0, j) = descriptors[j];
        }
        float label = svm->predict(testDescriptor);
        String slabel;
        if (label > 0) {
            slabel = "masked";
        }
        else if (label < 0) {
            slabel = "unmasked";
        }
        cout << "This picture belongs to:" << slabel << endl;
        imshow("test image", image);

        waitKey();
    }
}


void train_svm_hog_descriptor() {
    String file_location = "Media/";
    vector<String> negativeLocations, positiveLocations;
    vector<Mat> negativeSamples, positiveSamples;
    glob(file_location + "Ground Truth/unmaskface_only", negativeLocations, false);
    glob(file_location + "Ground Truth/maskface_only", positiveLocations, false);

    size_t ncount = negativeLocations.size();
    size_t pcount = positiveLocations.size();

    cout << "negative samples size: " << ncount << "\n";
    cout << "positive samples size: " << pcount << "\n";

    for (size_t i = 0; i < ncount; i++) {
        negativeSamples.push_back(imread(negativeLocations[i]));
    }
    for (size_t i = 0; i < pcount; i++) {
        positiveSamples.push_back(imread(positiveLocations[i]));
    }
    ncount = 5000;
    pcount = 5000; 
    Ptr<Boost> boost = Boost::create();
    
    HOGDescriptor hog(Size(50, 50), Size(10, 10), Size(5, 5), Size(10, 10),
        9, 1, -1, HOGDescriptor::L2Hys, 0.2,
        false, HOGDescriptor::DEFAULT_NLEVELS, false);
    //HOGDescriptor hog(Size(50, 50), Size(10, 10), Size(5, 5), Size(10, 10), 9);
    int descriptorDimension;
    Mat sampleFeatureMat, sampleLabelMat;

    for (int i = 0; i < pcount; i++) {
        cout << i << "\n";
        vector<float> descriptors;
        Mat image;
        resize(positiveSamples[i], image, Size(100, 100));
        hog.compute(image, descriptors, Size(8, 8));
        if (i == 0)
        {
            descriptorDimension = descriptors.size();
            sampleFeatureMat = Mat::zeros(pcount /*positive_samples*/ + ncount/*negative_samples*/,
                descriptorDimension, CV_32FC1);
            sampleLabelMat = Mat::zeros(pcount /*positive_samples*/ + ncount/*negative_samples*/, 1, CV_32SC1);
        }
        for (int j = 0; j < descriptorDimension; j++)
        {
            
            sampleFeatureMat.at<float>(i, j) = descriptors[j];
        }
        sampleLabelMat.at<float>(i, 0) = 1;
    }
    for (int i = 0; i < ncount; i++)
    {
        cout << i << "\n";
        vector<float> descriptors;
        Mat image;
        resize(negativeSamples[i], image, Size(100, 100));
        hog.compute(image, descriptors, Size(8, 8));

        for (int j = 0; j < descriptorDimension; j++)
        {
            sampleFeatureMat.at<float>(i + pcount/*positive_samples*/, j) = descriptors[j];
        }
        sampleLabelMat.at<float>(i + pcount /*positive_samples*/, 0) = -1;
    }
    Ptr<TrainData> td = TrainData::create(sampleFeatureMat, SampleTypes::ROW_SAMPLE, sampleLabelMat);
    boost->train(td);
    boost->save("ADABOOST_TEST_3.xml");
    return;
}