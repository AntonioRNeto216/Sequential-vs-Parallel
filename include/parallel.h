#ifndef PARALLEL_H
#define PARALLEL_H

#include <vector>
#include <opencv2/opencv.hpp>

#include <defines.h>

void *h1ThreadFunction(
    void *arg
);

void *h2ThreadFunction(
    void *arg
);

void performParallelMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters,
    int *numThreads
);

#endif