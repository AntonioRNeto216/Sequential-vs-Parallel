#ifndef DEFINES_H
#define DEFINES_H

#include <opencv2/opencv.hpp>

enum Method
{
    SEQUENTIAL,
    PARALLEL,
    PARALLEL_OPEN_MP
};

typedef struct Filters_s
{
    cv::Mat h1 = (cv::Mat_<float>(3,3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    cv::Mat h2 = (cv::Mat_<float>(3,3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
} Filters_t;

typedef struct PerformH1ParallelParameters_s
{
    std::vector<cv::Mat> *input;
    std::vector<cv::Mat> *result;
    cv::Mat *filter;
    int *index;
    pthread_mutex_t *mutex_index;
    std::queue<int> *indexes_to_process;
    pthread_mutex_t *mutex_indexes_to_process;
} PerformH1ParallelParameters_t;

typedef struct PerformH2ParallelParameters_s
{
    std::vector<cv::Mat> *input;
    std::vector<cv::Mat> *result;
    cv::Mat *filter;
    std::queue<int> *indexes_to_process;
    pthread_mutex_t *mutex_indexes_to_process;
    int *counter_frames_processed;
} PerformH2ParallelParameters_t;

#endif