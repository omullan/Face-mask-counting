#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>


int main(int argc, char* argv[])
{
    String file_location = "Media/";
    VideoCapture capture(file_location + "Face Masks KDH 1.avi");
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open" << endl;
        return 0;
    }
   
    int frameCount = capture.get(CAP_PROP_FRAME_COUNT);
    

    run();

    //Mat* frames = runWithoutBM(capture);
    /*
    writeVideoToFile(frames, "finalclassfier.avi", 30, frames[0].size().width, frames[0].size().height, frameCount-5);
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
    boost = StatModel::load<Boost>("ADABOOST_TEST_4.xml");
    if (boost->empty()) {
        cout << "could not load model";
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
        /*
        if (faceDetectCounter == 5 || firstFaceDetected) {
            vector<String> result;
            Mat tmp = detectMaskedFaces(foreground, net, result, boost);
            if (!tmp.empty()) {
                faceDetect = tmp;
            }   
            faceDetectCounter = 0;
        }
        */
        vector<String> result;
        vector<Rect> faces;
        faceDetect = detectMaskedFaces(foreground, net, result, boost, faces);
        vector<Mat> vec = { frame, cleanedImage, foreground, faceDetect };
        
        Mat out = makeCanvas(vec, 600, 1);
        /*
        imshow("", out);
        char c = waitKey(1);
        cout << out.size();
        */
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
    vector<vector<int>> videoGT = readCSV("video GT.csv");
    Ptr<Boost> boost = Boost::create();
    boost = StatModel::load<Boost>("ADABOOST_TEST_3.xml");
    if (boost->empty()) {
        cout << "could not load model";
        return masks;
    }
    video >> frame;
    int faceDetectCounter = 0;
    int GTcount = 0;
    bool isRelevant = false;
    Mat faceDetect = frame.clone();
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    
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
        faceDetect = detectMaskedFaces(frame, net, result, boost, faces);
        if (result.size() != 0) {
            cout << "frame : " << i << " Result: " << result[0] << "\n";
        }


        /*
        
        if (i == videoGT[GTcount][0] || i == videoGT[GTcount][1]) {
            isRelevant = !isRelevant;
            GTcount++;
        }

        if (isRelevant) {
            if (videoGT[GTcount][1] == 0) {
                if (result.size() != 0) {
                    if (result[0] == "Masked") {
                        tp++;
                    }
                    else {
                        fn++;
                    }
                }
                else {
                    fn++;
                }
            }
            else if(videoGT[GTcount][0] == 0) {
                if (result.size() != 0) {
                    if (result[0] == "Unmasked") {
                        tn++;
                    }
                    else {
                        fp++;
                    }
                }
                else {
                    fn++;
                }
            }
        }
        */
        
        vector<Mat> vec = { frame, faceDetect };
        Mat out = makeCanvas(vec, 600, 2);
        /*
        imshow("", out);
        char c = waitKey(1);
        */
        masks[i] = out;
        /*
        cout << out.size();
        
        cout << "frame: " << i << "\n";
        */
        video >> frame;
    }

    return masks;
}
map<int, vector<int>> readMAFAGT() {
    map<int, vector<int>> groundTruth;
    String filename = "mafatestlabel.csv";
    ifstream myfile("Media/Ground Truth/" + filename);
    String line;
    if (!myfile.is_open()) throw runtime_error("Could not open file");
    while (getline(myfile, line))
    {
        istringstream s(line);
        string field;
        int count = 0;
        int key;
        vector<int> vec;
        vec.clear();
        while (getline(s, field, ',')) {
            if (count == 0) {
                String tmp = field.substr(5, 8);
                stringstream ss(tmp);
                int x;
                ss >> x;
                key = x;
            }
            else if (!field.empty()) {
                stringstream ss(field);
                int x;
                ss >> x;
                vec.push_back(x);
            }
           
            count++;
        }
        int numberOfFaces = vec.size() / 18;
        if (numberOfFaces > 1) {
            vector<int> output;
            for (int i = 0; i < numberOfFaces; i++) {
                for (int j = 0; j < vec.size(); j = j + numberOfFaces) {
                    output.push_back(vec[j + i]);
                }
            }
            groundTruth.insert(pair<int, vector<int>>(key, output));
        }
        else {
            groundTruth.insert(pair<int, vector<int>>(key, vec));
        }
    }
    return groundTruth;
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
    ncount = 10000;
    pcount = 10000; 
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
    boost->save("ADABOOST_TEST_4.xml");
    return;
}

vector<vector<int>> readCSV(String filename) {
    string line, entry;
    vector<String> result;
    vector<vector<int>> out;
    ifstream myfile("Media/Ground Truth/" + filename);
    if (!myfile.is_open()) throw runtime_error("Could not open file");

    if (myfile.good()) {
        getline(myfile, line);
        stringstream ss(line);
        while (getline(ss, entry, ',')) {

            result.push_back(entry);
        }
    }

    if (result.size() > 20) {
        for (int i = 0; i < result.size(); i++) {
            if (result[i].length() == 2) {
                if (result[i][1] == 'M') {
                    vector<int> vec = { result[i][0] - '0', 0 };
                    out.push_back(vec);
                }
                else {
                    vector<int> vec = { 0, result[i][0] - '0' };
                    out.push_back(vec);
                }
            }
            else {
                vector<int> vec = { result[i][0] - '0', result[i][3] - '0' };
                out.push_back(vec);
            }
        }
    }
    else {
        for (int i = 0; i < result.size(); i++) {
            if (result[i].size() == 4) {
                int output = 0;
                output += (result[i][0] - '0') * 100;
                output += (result[i][1] - '0') * 10;
                output += (result[i][2] - '0');
                if (result[i][3] == 'M') {
                    vector<int> vec = { output, 0 };
                    out.push_back(vec);
                }
                else {
                    vector<int> vec = { 0, output };
                    out.push_back(vec);
                }
            }
            else {
                int output = 0;
                output += (result[i][0] - '0') * 1000;
                output += (result[i][1] - '0') * 100;
                output += (result[i][2] - '0') * 10;
                output += (result[i][3] - '0');
                if (result[i][4] == 'M') {
                    vector<int> vec = { output, 0 };
                    out.push_back(vec);
                }
                else {
                    vector<int> vec = { 0, output };
                    out.push_back(vec);
                }
            }
        }
    }
    
    return out;
}

