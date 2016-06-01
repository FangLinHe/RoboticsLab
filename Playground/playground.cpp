#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const float PI = 3.14159265;
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);
cv::Point2f lineIntersection(cv::Point &p1, cv::Point &p2, cv::Point &p3, cv::Point &p4);
char key;
int main() {
  VideoCapture cap(0);
  if(!cap.isOpened())  // check if we succeeded
    return -1;
  
  // Camera Calibration
  Mat cameraMatrix, distCoeffs;
  FileStorage fs("C:\\out_camera_data.xml", FileStorage::READ);
  if (fs.isOpened()) {
    fs["Camera_Matrix"] >> cameraMatrix;
    fs["Distortion_Coefficients"] >> distCoeffs;
    fs.release();
  }

  system("pause");

  for(;;) {
    Mat temp,frame;
    cap >> temp; // get a new frame from camera
    cv::undistort(temp, frame, cameraMatrix, distCoeffs);
    
    // parameters
    const int TAU = 3;
    const int SIG_L = 200;
    const int SIG_D = 20;
    const int B = 2;
    const float SIG_H = 0.0007;
    const float THRES = 20;
    cv::Mat image = frame.clone();

    // court points

    // convert to gray scale
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, CV_BGR2GRAY);

    // white pixel detection
    cv::Mat white_pixels = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    for (int i = TAU; i < image.rows-TAU; ++i) {
      for (int j = TAU; j < image.cols-TAU; ++j) {
        if ((image_gray.at<uchar>(i, j) <= SIG_L &&
             image_gray.at<uchar>(i-TAU, j) - image_gray.at<uchar>(i, j) > SIG_D &&
             image_gray.at<uchar>(i+TAU, j) - image_gray.at<uchar>(i, j) > SIG_D) ||
             (image_gray.at<uchar>(i, j) <= SIG_L &&
             image_gray.at<uchar>(i, j-TAU) - image_gray.at<uchar>(i, j) > SIG_D &&
             image_gray.at<uchar>(i, j+TAU) - image_gray.at<uchar>(i, j) > SIG_D)
         ) {
          white_pixels.at<uchar>(i, j) = 255;
        }
      }
    }
  
    cv::imshow("white pixels", white_pixels);

    // compute structure matrix within the pixel neighborhood
    // gradient x
    cv::Mat image_gradient_x;
    cv::Sobel(image_gray, image_gradient_x, image_gray.depth(), 1, 0);
    cv::convertScaleAbs(image_gradient_x, image_gradient_x);

    // gradient y
    cv::Mat image_gradient_y;
    cv::Sobel(image_gray, image_gradient_y, image_gray.depth(), 0, 1);
    cv::convertScaleAbs(image_gradient_y, image_gradient_y);

    // Apply Hough transform to find the lines
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(white_pixels, lines, 1, CV_PI/180, 120);

    // group the lines according to the theta
    std::vector<std::vector<int>> groups(lines.size(), std::vector<int>(lines.size()));
    for (int i = 0; i < lines.size(); ++i)
      for (int j = 0; j < lines.size(); ++j)
        groups[i][j] = -1;

    if (lines.size() > 0)
      groups[0][0] = 0;

    int num_groups = 1;
    for (size_t i = 1; i < lines.size(); ++i) {
      float min_distance = image.rows*image.cols;
      int min_index = 0;
      for (size_t ii = 0; ii < num_groups; ++ii) {
        int line_index = groups[ii][0];
        float distance_rho = std::abs(lines[i][0] - lines[line_index][0]);
        float distance_theta = std::abs(lines[i][1] - lines[line_index][1]);
        float distance = distance_theta * 180 / PI + distance_rho;
        if (distance < min_distance) {
            min_distance = distance;
            min_index = ii;
        }
      }

      // not belong to any groups, add one group
      if (min_distance > THRES) {
        num_groups++;
        groups[num_groups-1][0] = i;
      }
      else {// append to the end
        for (size_t ii = 0; ii < lines.size(); ++ii) {
          if (groups[min_index][ii] == -1) {
            groups[min_index][ii] = i;
            break;
          }
        }
      }
    }

    cv::Mat color_dst2; // for drawing lines
    cv::cvtColor(white_pixels, color_dst2, CV_GRAY2BGR);
    std::vector<std::vector<float>> refine_lines(num_groups, 3); // 0: rho, 1: theta, 3: 1 for horizontal and -1 for vertical
    for (int i = 0; i < num_groups; ++i) {
      float sum_rho = 0;
      float sum_theta = 0;
      int count = 0;
      for (int j = 0; j < lines.size(); ++j) {
        int line_index = groups[i][j];
        if (line_index != -1) {
          count++;
          sum_rho += lines[line_index][0];
          sum_theta += lines[line_index][1];
        }
        else
          break;
      }
      
      float rho = sum_rho / (float)count;
      float theta = sum_theta / (float)count;
      refine_lines[i][0] = rho;
      refine_lines[i][1] = theta;
      if (theta > PI*(1.0/3.0) && theta < PI*(2.0/3.0))
        refine_lines[i][2] = 1;
      else
        refine_lines[i][2] = -1;

      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      cv::Point pt1(cvRound(x0 + 1000*(-b)),
          cvRound(y0 + 1000*(a)));
      cv::Point pt2(cvRound(x0 - 1000*(-b)),
          cvRound(y0 - 1000*(a)));

      if (refine_lines[i][2] > 0) // horizontal
        line(color_dst2, pt1, pt2, cv::Scalar(0,0,255), 1, 8);
      else // vertical
        line(color_dst2, pt1, pt2, cv::Scalar(0,255,0), 1, 8);
    }

    // Get the min and max two vertical lines &
    // the min 2 horizontal lines as the reference lines
    float min_ver_rho = 1000000;
    float max_ver_rho = -1000000;
    int min_ver_index = 0;
    int max_ver_index = 0;
    float max_hor_rho_1 = -1000000;
    float max_hor_rho_2 = -1000001;
    int max_hor_index_1 = 0;
    int max_hor_index_2 = 0;
    for (int i = 0; i < num_groups; ++i) {
      float rho = refine_lines[i][0];
      if (refine_lines[i][2] > 0) { // horizontal
        if (rho > max_hor_rho_1) {
          max_hor_rho_2 = max_hor_rho_1;
          max_hor_rho_1 = rho;
          max_hor_index_2 = max_hor_index_1;
          max_hor_index_1 = i;
        }
        else if (rho > max_hor_rho_2) {
          max_hor_rho_2 = rho;
          max_hor_index_2 = i;
        }
      }
      else { // vertical
        if (rho < min_ver_rho) {
          min_ver_rho = rho;
          min_ver_index = i;
        }
        if (rho > max_ver_rho) {
          max_ver_rho = rho;
          max_ver_index = i;
        }
      }
    }
    int indices[4];
    indices[0] = max_hor_index_1;
    indices[1] = max_hor_index_2;
    indices[2] = min_ver_index;
    indices[3] = max_ver_index;
    std::vector<std::vector<cv::Point>> points(4, 2);


    for (int i = 0; i < 4; ++i) {
      int index = indices[i];
      float rho = refine_lines[index][0];
      float theta = refine_lines[index][1];
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      points[i][0] = cv::Point(cvRound(x0 + 1000*(-b)),
          cvRound(y0 + 1000*(a)));
      points[i][1] = cv::Point(cvRound(x0 - 1000*(-b)),
          cvRound(y0 - 1000*(a)));
    }

    cv::Point2f p1 = lineIntersection(points[0][0], points[0][1], points[2][0], points[2][1]);
    cv::circle(color_dst2, p1, 3, cv::Scalar(255, 0, 0), 3);
    cv::Point2f p2 = lineIntersection(points[0][0], points[0][1], points[3][0], points[3][1]);
    cv::circle(color_dst2, p2, 3, cv::Scalar(0, 255, 0), 3);
    cv::Point2f p3 = lineIntersection(points[1][0], points[1][1], points[2][0], points[2][1]);
    cv::circle(color_dst2, p3, 3, cv::Scalar(0, 0, 255), 3);
    cv::Point2f p4 = lineIntersection(points[1][0], points[1][1], points[3][0], points[3][1]);
    cv::circle(color_dst2, p4, 3, cv::Scalar(255, 255, 255), 3);
    
    imshow("color dst 2", color_dst2);

    std::vector<cv::Point2f> point_array_src(4);
    point_array_src[0] = p1;
    point_array_src[1] = p2;
    point_array_src[2] = p3;
    point_array_src[3] = p4;
    std::vector<cv::Point2f> point_array_dst(4);
    point_array_dst[0] = cv::Point2f(530, 370);
    point_array_dst[1] = cv::Point2f(10, 370);
    point_array_dst[2] = cv::Point2f(530, 10);
    point_array_dst[3] = cv::Point2f(10, 10);
    cv::Mat homography1 = cv::findHomography(point_array_src, point_array_dst);
    cv::Mat homography2 = cv::findHomography(point_array_dst, point_array_src);
    point_array_dst[0] = cv::Point2f(26, 18);
    point_array_dst[1] = cv::Point2f(0, 18);
    point_array_dst[2] = cv::Point2f(26, 0);
    point_array_dst[3] = cv::Point2f(0, 0);
    cv::Mat homography3 = cv::findHomography(point_array_src, point_array_dst);
    cv::Mat image_32fc3;
    image.convertTo(image_32fc3, CV_32FC3);
    cv::Mat transform_image;
    cv::warpPerspective(image, transform_image, homography1, cv::Size(540, 380));
    imshow("transform", transform_image);

    // volleyball model
    std::vector<std::vector<cv::Point2f>> line_segments(4, 2);
    line_segments[0][0] = cv::Point2f(10, 10);
    line_segments[0][1] = cv::Point2f(10, 370);
    line_segments[1][0] = cv::Point2f(10, 370);
    line_segments[1][1] = cv::Point2f(530, 370);
    line_segments[2][0] = cv::Point2f(530, 370);
    line_segments[2][1] = cv::Point2f(530, 10);
    line_segments[3][0] = cv::Point2f(530, 10);
    line_segments[3][1] = cv::Point2f(10, 10);

    cv::Mat color_dst3 = image.clone();

    std::vector<cv::Point2f> middle_line(2);
    for (int i = 0; i < line_segments.size(); ++i) {
      std::vector<cv::Point2f> src_points(2);
      std::vector<cv::Point2f> dst_points(2);
      src_points[0] = line_segments[i][0]; 
      src_points[1] = line_segments[i][1];
      perspectiveTransform(src_points, dst_points, homography2);
      cv::line(color_dst3, dst_points[0], dst_points[1], cv::Scalar(0, 255/7*(i+1), 0), 3, 4);
      if (i == 4) {
        middle_line[0] = dst_points[0];
        middle_line[1] = dst_points[1];
      }
    }


    Mat framehsv, img_violet;
        cvtColor(frame, framehsv, CV_BGR2HSV);
    Mat channels[3];
    split(frame, channels);
    cv::inRange(framehsv, cv::Scalar(147,92,35), cv::Scalar(171,230,173), img_violet);
    
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                                        Size(5, 5),
                                        Point(3, 3));
    /// Apply the dilation operation
    erode(img_violet, img_violet, element);
    element = getStructuringElement(MORPH_ELLIPSE,
                                    Size(5, 5),
                                    Point(3, 3));
    dilate(img_violet, img_violet, element);

    cv::Mat out;
    Mat g = Mat::zeros(Size(frame.rows, frame.cols), CV_8UC1);
    cv::Mat in[] = {img_violet, img_violet, img_violet};
    cv::merge(in, 3, out);
    cv::addWeighted(color_dst3, 1, out, 1, 0.0, out);

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

    vector<Point2f> scene_points, mapped_coord;
    scene_points.push_back(cv::Point2f(sumj/count,sumi/count));

    perspectiveTransform(scene_points, mapped_coord, homography3);
    cout << mapped_coord[0].x << ", " << mapped_coord[0].y << endl;

    imshow("court", out);

    if (waitKey(30) >= 0)
      break;
  }

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}


cv::Point2f lineIntersection(cv::Point &p1, cv::Point &p2, cv::Point &p3, cv::Point &p4) {
  float bx = p2.x - p1.x;
  float by = p2.y - p1.y;
  float dx = p4.x - p3.x;
  float dy = p4.y - p3.y; 
  float b_dot_d_perp = bx*dy - by*dx;
  if (b_dot_d_perp == 0) {
    return NULL;
  }
  float cx = p3.x - p1.x; 
  float cy = p3.y - p1.y;
  float t = (cx*dy - cy*dx) / b_dot_d_perp; 

  return cv::Point2f(p1.x+t*bx, p1.y+t*by); 
}

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
  Point2f x = o2 - o1;
  Point2f d1 = p1 - o1;
  Point2f d2 = p2 - o2;

  float cross = d1.x*d2.y - d1.y*d2.x;
  if (abs(cross) < /*EPS*/1e-8)
    return false;

  double t1 = (x.x * d2.y - x.y * d2.x)/cross;
  r = o1 + d1 * t1;
  return true;
}