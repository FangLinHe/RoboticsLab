#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const float PI = 3.14159265;

char key;
int main() {
  VideoCapture cap(0);
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  namedWindow("Webcam", 1);
  system("pause");

  for (;;) {
    Mat frame, framehsv, img_violet;
    cap >> frame; // get a new frame from camera
    cvtColor(frame, framehsv, CV_BGR2HSV);
    Mat channels[3];
    split(frame, channels);
    cv::inRange(framehsv, cv::Scalar(147,92,35), cv::Scalar(171,230,173), img_violet);
    
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                                        Size(11, 11),
                                        Point(5, 5));
    /// Apply the dilation operation
    erode(img_violet, img_violet, element);
    element = getStructuringElement(MORPH_ELLIPSE,
                                    Size(11, 11),
                                    Point(7, 7));
    dilate(img_violet, img_violet, element);

    cv::Mat out;
    Mat g = Mat::zeros(Size(frame.rows, frame.cols), CV_8UC1);
    cv::Mat in[] = {img_violet, img_violet, img_violet};
    cv::merge(in, 3, out);
    cv::addWeighted(frame, 1, out, 1, 0.0, out);

    double sum = 0, sumi = 0, sumj = 0;
    double count = 0;
    for (double i = 0; i < img_violet.rows; i+=1.)
      for (double j = 0; j < img_violet.cols; j+=1.)
        if (img_violet.at<bool>(i, j)) {
          sum++;
          count += 1.;
          sumi += i;
          sumj += j;
        }

    cv::Mat violet_masked;
    cv::circle(out, cv::Point((int)(sumj/count),(int)(sumi/count)), sqrt(sum/PI), Scalar(0,255,0));
    cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    cv::circle(mask, cv::Point((int)(sumj/count),(int)(sumi/count)), sqrt(sum/PI), Scalar(255,255,255), -1, 8, 0); //-1 means filled
    img_violet.copyTo(violet_masked, mask); // copy values of img to dst if mask is > 0.
    if (cv::sum(img_violet).val[0] != 0) {
      double jaccard = std::abs((cv::sum(violet_masked)/cv::sum(img_violet)).val[0]);
      double violet_area_percentage = cv::sum(img_violet).val[0]/((double)frame.rows*(double)frame.cols*255);
      double punish_term = 1/(1+std::exp(-(violet_area_percentage-0.005)*500));
      double conf_ind = jaccard * punish_term;
      cout << "Confidence value: " << conf_ind << endl;
      // The confidence value is calculated by the Jaccard index times the punishment term.
      // Punishment term is determined by the total area detected in the environment.
      // TODO: to calculate the variance (how the violet areas are distributed) as another punishment term.
    }

    if (!frame.empty())
      imshow("Webcam", out);

    if(waitKey(30) >= 0) break;
  }

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}