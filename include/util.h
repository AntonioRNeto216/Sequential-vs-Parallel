#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <opencv2/opencv.hpp>

#include <defines.h>

void performFilter(
    cv::Mat *input,
    cv::Mat *result,
    cv::Mat *filter,
    int rows,
    int cols
);

void getFrames(
    std::vector<cv::Mat> *all_video_frames,
    cv::VideoCapture *videoCapture,
    std::string *video_path
);

void playPerformedVideo(
    std::vector<cv::Mat> *all_video_frames
);

int videoMenu();

Method methodMenu();

int numThreadsMenu();

#endif