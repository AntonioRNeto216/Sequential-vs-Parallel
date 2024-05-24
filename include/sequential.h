#ifndef SEQUENTIAL_METHOD_H
#define SEQUENTIAL_METHOD_H

#include <vector>
#include <opencv2/opencv.hpp>

#include <defines.h>

void performSequencialMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters
);

#endif