#ifndef PARALLEL_OPEN_MP_H
#define PARALLEL_OPEN_MP_H

#include <omp.h>

#include <defines.h>

void performFilterOpenMP(
    cv::Mat *input,
    cv::Mat *result,
    cv::Mat *filter,
    int rows,
    int cols
);

void performParallelOpenMPMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters,
    int *numThreads
);

#endif