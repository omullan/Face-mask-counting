/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe � Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include "opencv2/video.hpp"

void drawOpticalFlow(Mat& optical_flow, Mat& display, int spacing, Scalar passed_line_colour = -1.0, Scalar passed_point_colour = -1.0)
{
	Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
	for (int row = spacing / 2; row < display.rows; row += spacing)
		for (int column = spacing / 2; column < display.cols; column += spacing)
		{
			const Point2f& fxy = optical_flow.at<Point2f>(row, column);
			circle(display, Point(column, row), 1, (passed_point_colour.val[0] == -1.0) ? colour : passed_point_colour, -1);
			line(display, Point(column, row), Point(cvRound(column + fxy.x), cvRound(row + fxy.y)),
				(passed_line_colour.val[0] == -1.0) ? colour : passed_line_colour);
		}
}

#define MAX_FEATURES 400
void LucasKanadeOpticalFlow(Mat& previous_gray_frame, Mat& gray_frame, Mat& display_image)
{
	Size img_sz = previous_gray_frame.size();
	int win_size = 10;
	cvtColor(previous_gray_frame, display_image, COLOR_GRAY2BGR);
	vector<Point2f> previous_features, current_features;
	const int MAX_CORNERS = 500;
	goodFeaturesToTrack(previous_gray_frame, previous_features, MAX_CORNERS, 0.05, 5, noArray(), 3, false, 0.04);
	cornerSubPix(previous_gray_frame, previous_features, Size(win_size, win_size), Size(-1, -1),
		TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.03));
	vector<uchar> features_found;
	calcOpticalFlowPyrLK(previous_gray_frame, gray_frame, previous_features, current_features, features_found, noArray(),
		Size(win_size * 4 + 1, win_size * 4 + 1), 5,
		TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, .3));
	for (int i = 0; i < (int)previous_features.size(); i++)
	{
		if (!features_found[i])
			continue;
		circle(display_image, previous_features[i], 1, Scalar(0, 0, 255));
		line(display_image, previous_features[i], current_features[i], Scalar(0, 255, 0));
	}
}

class MedianBackground
{
private:
	Mat mMedianBackground;
	float**** mHistogram;
	float*** mLessThanMedian;
	float mAgingRate;
	float mCurrentAge;
	float mTotalAges;
	int mValuesPerBin;
	int mNumberOfBins;
public:
	MedianBackground(Mat initial_image, float aging_rate, int values_per_bin);
	Mat GetBackgroundImage();
	void UpdateBackground(Mat current_frame);
	float getAgingRate()
	{
		return mAgingRate;
	}
};

MedianBackground::MedianBackground(Mat initial_image, float aging_rate, int values_per_bin)
{
	mCurrentAge = 1.0;
	mAgingRate = aging_rate;
	mTotalAges = 0.0;
	mValuesPerBin = values_per_bin;
	mNumberOfBins = 256 / mValuesPerBin;
	mMedianBackground = Mat::zeros(initial_image.size(), initial_image.type());
	mLessThanMedian = (float***) new float** [mMedianBackground.rows];
	mHistogram = (float****) new float*** [mMedianBackground.rows];
	for (int row = 0; (row < mMedianBackground.rows); row++)
	{
		mHistogram[row] = (float***) new float** [mMedianBackground.cols];
		mLessThanMedian[row] = (float**) new float* [mMedianBackground.cols];
		for (int col = 0; (col < mMedianBackground.cols); col++)
		{
			mHistogram[row][col] = (float**) new float* [mMedianBackground.channels()];
			mLessThanMedian[row][col] = new float[mMedianBackground.channels()];
			for (int ch = 0; (ch < mMedianBackground.channels()); ch++)
			{
				mHistogram[row][col][ch] = new float[mNumberOfBins];
				mLessThanMedian[row][col][ch] = 0.0;
				for (int bin = 0; (bin < mNumberOfBins); bin++)
				{
					mHistogram[row][col][ch][bin] = (float)0.0;
				}
			}
		}
	}
}

Mat MedianBackground::GetBackgroundImage()
{
	return mMedianBackground;
}

void MedianBackground::UpdateBackground(Mat current_frame)
{
	mTotalAges += mCurrentAge;
	float total_divided_by_2 = mTotalAges / ((float)2.0);
	for (int row = 0; (row < mMedianBackground.rows); row++)
	{
		for (int col = 0; (col < mMedianBackground.cols); col++)
		{
			for (int ch = 0; (ch < mMedianBackground.channels()); ch++)
			{
				int new_value = (mMedianBackground.channels() == 3) ? current_frame.at<Vec3b>(row, col)[ch] : current_frame.at<uchar>(row, col);
				int median = (mMedianBackground.channels() == 3) ? mMedianBackground.at<Vec3b>(row, col)[ch] : mMedianBackground.at<uchar>(row, col);
				int bin = new_value / mValuesPerBin;
				mHistogram[row][col][ch][bin] += mCurrentAge;
				if (new_value < median)
					mLessThanMedian[row][col][ch] += mCurrentAge;
				int median_bin = median / mValuesPerBin;
				while ((mLessThanMedian[row][col][ch] + mHistogram[row][col][ch][median_bin] < total_divided_by_2) && (median_bin < 255))
				{
					mLessThanMedian[row][col][ch] += mHistogram[row][col][ch][median_bin];
					median_bin++;
				}
				while ((mLessThanMedian[row][col][ch] > total_divided_by_2) && (median_bin > 0))
				{
					median_bin--;
					mLessThanMedian[row][col][ch] -= mHistogram[row][col][ch][median_bin];
				}
				if (mMedianBackground.channels() == 3)
					mMedianBackground.at<Vec3b>(row, col)[ch] = median_bin * mValuesPerBin;
				else mMedianBackground.at<uchar>(row, col) = median_bin * mValuesPerBin;
			}
		}
	}
	mCurrentAge *= mAgingRate;
}

//My code
void runMedianBackground(VideoCapture video, float learningRate, int valuesPerBin) {
	Mat currentFrame;
	video >> currentFrame;
	Mat faceDetect = currentFrame.clone();
	MedianBackground medianBackground(currentFrame, learningRate, valuesPerBin);
	Mat medianBackgroundImage, medianForegroundImage;
	Net net = load();
	Ptr<Boost> boost = Boost::create();
	boost = StatModel::load<Boost>("ADABOOST_TEST_2.xml");
	if (boost->empty()) {
		cout << "could not load SVM";
		return;
	}
	int frameCount = 0;
	int faceDetectCounter = 0;
	while (!currentFrame.empty()) {
		medianBackground.UpdateBackground(currentFrame);
		medianBackgroundImage = medianBackground.GetBackgroundImage();
		Mat medianDifference;
		absdiff(medianBackgroundImage, currentFrame, medianDifference);
		cvtColor(medianDifference, medianDifference, COLOR_BGR2GRAY);
		threshold(medianDifference, medianDifference, 30, 255, THRESH_BINARY);
		medianForegroundImage.setTo(Scalar(0, 0, 0));
		currentFrame.copyTo(medianForegroundImage, medianDifference);

		String frameString = to_string(frameCount);
		if (faceDetectCounter == 10) {
			vector<String> result;
			vector<Rect> faces;
			faceDetect = detectMaskedFaces(medianForegroundImage, net, result ,boost, faces);
			faceDetectCounter = 0;
		}
		vector<Mat> vec = { currentFrame, medianBackgroundImage, medianForegroundImage, faceDetect };
		Mat out = makeCanvas(vec, 480, 2);

		imshow("", out);
		char c = waitKey(1);

		faceDetectCounter++;
		frameCount++;
		video >> currentFrame;
		cout << "Median: " << frameCount << "\n";
	}
}