#include <vector>
#include <opencv2/opencv.hpp>

#include <defines.h>

#ifndef SEQUENTIAL_METHOD_H
#define SEQUENTIAL_METHOD_H

void performSequencialMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters
);

#endif