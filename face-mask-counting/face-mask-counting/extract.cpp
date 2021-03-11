#include "Utilities.h"


//code to extract faces from masked images training set, for the purpose of training classifier
void extract() {
    String file_location = "Media/Ground Truth/Datasets/";
    vector<String> fn;
    glob(file_location + "lfw/1", fn, false);
    vector<Mat> images;
    size_t count = fn.size();
    Net net = load();
    cout << count + "\n";
    int face_count = 0;
    float confidenceThreshold = 0.5;
    for (size_t i = 0; i < count; i++) {
        Mat tmp = imread(fn[i]);
        vector<Rect> faces;
        DNNfaceDetect(net, tmp, faces, confidenceThreshold);
        for (int j = 0; j < faces.size(); j++) {
            face_count++;
            imwrite("Media/Ground Truth/unmaskface_only/facetrain" + to_string(face_count) + ".png", tmp(faces[j]));
            cout << face_count << "\n";
        }
    }
}