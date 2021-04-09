#include "Utilities.h"

void startup() {
    String file_location = "Media/";
    vector<String> fn;
    glob(file_location + "Ground Truth/singlefacetesting", fn, false);
    vector<Mat> images;
    size_t count = fn.size();

    cout << count + "\n";
    for (size_t i = 0; i < count; i++) {
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

    if (images.size() == 200) {
        for (int i = 0; i < images.size(); i++) {
            vector<String> result;
            vector<Rect> faces;
            Mat out = detectMaskedFaces(images[i], net, result, boost, faces);
            if (result.size() != 0) {
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
        cout << "\n" << "True Negative rate unmasked: " << unmaskedCorrectCount;
        cout << "\n" << "False positive rate unmasked: " << (100 - unmaskedCorrectCount);
        cout << "\n" << "Unasked faces not found: " << unmaskedFaceNotFoundCount << "\n";
    }
}


Mat findOptimumWeighting(Mat image, Net net, vector<String>& result, Ptr<Boost> boost, float spWeight, float histWeight, float boostWeight) {
    vector<Rect> faces;
    int width, height, x, y;
    float confidenceThreshold = 0.5;
    DNNfaceDetect(net, image, faces, confidenceThreshold);
    if (faces.size() != 0) {
        for (int i = 0; i < faces.size(); i++) {
            Mat haarImage = image(faces[i]);
            width = faces[i].width; height = faces[i].height; x = faces[i].x; y = faces[i].y;
            Rect topHalfFace(x, y, width, height / 2);
            Rect bottomHalfFace(x, y + (height / 2), width, height / 2);

            double histMatchingScore = faceHistogram(image, topHalfFace, bottomHalfFace);
            Mat colourThresholdSkin = detectSkin(image);
            vector<double> skinProbablities = countPixels(colourThresholdSkin, topHalfFace, bottomHalfFace);
            bool is_mask = boosted_mask_classifier(boost, haarImage);

            double unmaskScore = 0.0;
            if (!is_mask) {
                unmaskScore += boostWeight;
            }
            if (histMatchingScore > 0.0 && histMatchingScore < 0.5) {
                //cout << "hist match: " << histMatchingScore;
                if (histMatchingScore <= 0.2) {
                    unmaskScore += histWeight;
                }
                else {
                    unmaskScore += (histMatchingScore - 0.2);
                }
            }
            if (skinProbablities[0] > 50 && skinProbablities[1] > 50) {
                float tmp = ((skinProbablities[0] + skinProbablities[1]) - 100) / 333.33;
                unmaskScore += (0.1 + tmp);
            }
            //cout << "Unmask score: " << unmaskScore << "\n";
            if (unmaskScore >= 0.5) {
                result.push_back("Unmasked");
            }
            else {
                result.push_back("Masked");
            }
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