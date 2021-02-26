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
    // Load Haar Cascade(s)
    vector<CascadeClassifier> cascades;
    String cascade_files[] = { "haarcascades/haarcascade_frontalface_alt2.xml",
                                "haarcascades/haarcascade_eye_tree_eyeglasses.xml"};
    int number_of_cascades = sizeof(cascade_files) / sizeof(cascade_files[0]);
    for (int cascade_file_no = 0; (cascade_file_no < number_of_cascades); cascade_file_no++)
    {
        CascadeClassifier cascade;
        string filename(file_location);
        filename.append(cascade_files[cascade_file_no]);
        if (!cascade.load(filename))
        {
            cout << "Cannot load cascade file: " << filename << endl;
            return -1;
        }
        else cascades.push_back(cascade);
    }
    int frameCount = capture.get(CAP_PROP_FRAME_COUNT);

    //run();
    Mat* frames = gaussianMixture(capture, cascades);
    
    writeVideoToFile(frames, "output3.avi", 30, 1044, 600, frameCount-5);
    for (int i = 0; i < (capture.get(CAP_PROP_FRAME_COUNT) - 5); i++) {
        imshow("", frames[i]);
        char c = waitKey(10);
    }

   
    //runMedianBackground(capture, (float)1.005, 1, cascades[0]);
}
Mat* gaussianMixture(VideoCapture video, vector<CascadeClassifier> cascades) {
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
    Mat skinSamples = imread("Media/SkinSamples.jpg");
    Net net = load();
    Ptr<Facemark> facemark = loadFacemarkModel();
    Mat element(2, 2, CV_8U, Scalar(1));
    Mat frame, mask;
    int frameCount = video.get(CAP_PROP_FRAME_COUNT);
    cout << frameCount;
    Mat* masks = new Mat[frameCount - 5];
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
            String result;
            Mat tmp = detectMaskedFaces(foreground, cascades, skinSamples, net, facemark, result);
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
