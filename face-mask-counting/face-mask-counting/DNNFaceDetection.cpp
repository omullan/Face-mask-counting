#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>

// code adapted from this source https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

Net load() {
	String file_location = "Media/haarcascades";
	String caffeConfigFile = file_location + "/deploy.prototxt";
	String caffeWeightFile = file_location + "/res10_300x300_ssd_iter_140000.caffemodel";
	Net net = readNetFromCaffe(caffeConfigFile, caffeWeightFile);
	return net;
 }

void DNNfaceDetect(Net net, Mat image, vector<Rect> &faces, float confidenceThreshold)  {
	faces.clear();
	Mat out;

	if (!image.empty()) {
		Size imageSize = image.size();
		cout << imageSize.height + " :" << imageSize.width + "\n";
		Mat inputBlob = blobFromImage(image, 1.0, Size(300, 300), Scalar(104, 117, 123), false, false);

		net.setInput(inputBlob, "data");

		Mat detection = net.forward("detection_out");
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * imageSize.width);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * imageSize.height);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * imageSize.width);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * imageSize.height);
				Rect face(x1, y1, (abs(x2 - x1)), (abs(y2 - y1)));
				faces.push_back(face);
			}
		}
	}
}