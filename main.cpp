#include <iostream>
#include <vector>
#include <pthread.h>
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

typedef struct PerformFilterParameters_s
{
    cv::Mat *input;
    cv::Mat *result;
    cv::Mat *filter;
    int rows;
    int cols;
} PerformFilterParameters_t;

void initializePerformFilterParametersValues(
    PerformFilterParameters_t *performFiltersParameters,
    cv::Mat *input,
    cv::Mat *result,
    cv::Mat *filter,
    int rows,
    int cols
)
{
    performFiltersParameters->input = input; 
    performFiltersParameters->result = result;
    performFiltersParameters->filter = filter;
    performFiltersParameters->rows = rows;
    performFiltersParameters->cols = cols;
}

void performFilterSequencial(
    PerformFilterParameters_t *performFiltersParameters
)
{
    int mm, nn, ii, jj;
    float pixelResult;

    for (int i = 0; i < performFiltersParameters->rows; ++i)
    {
        for (int j = 0; j < performFiltersParameters->cols; ++j)
        {
            pixelResult = 0.0;

            for (int ik = 0; ik < 3; ++ik)
            {
                mm = 2 - ik;

                for (int jk = 0; jk < 3; ++jk)
                {
                    nn = 2 - jk;
                    
                    ii = i + (1 - mm);
                    jj = j + (1 - nn);

                    if (ii >= 0 && ii < performFiltersParameters->rows && jj >= 0 && jj < performFiltersParameters->cols)
                    {
                        pixelResult += performFiltersParameters->input->at<uchar>(ii, jj) * performFiltersParameters->filter->at<float>(mm, nn);
                    }

                }
            }

            performFiltersParameters->result->at<uchar>(i, j) = cv::saturate_cast<uchar>(pixelResult);
        }
    }
}

cv::Mat performSequencialMethod(
    cv::Mat *currentFrame,
    Filters_t *filters
)
{
    int rows = currentFrame->rows;
    int cols = currentFrame->cols;
    
    // Perform h1 filter

    cv::Mat h1Results = cv::Mat::zeros(
        currentFrame->size(), 
        currentFrame->type()
    );
    
    PerformFilterParameters_t performFiltersParameters_H1;
    initializePerformFilterParametersValues(
        &performFiltersParameters_H1, 
        currentFrame, 
        &h1Results, 
        &filters->h1, 
        rows, 
        cols
    );
    
    performFilterSequencial(&performFiltersParameters_H1);

    // Perform h2 filter

    cv::Mat h2Results = cv::Mat::zeros(
        performFiltersParameters_H1.result->size(), 
        performFiltersParameters_H1.result->type()
    );

    PerformFilterParameters_t performFiltersParameters_H2;
    initializePerformFilterParametersValues(
        &performFiltersParameters_H2, 
        performFiltersParameters_H1.result, 
        &h2Results, 
        &filters->h2, 
        rows, 
        cols
    );

    performFilterSequencial(&performFiltersParameters_H2);
    
    return h2Results;
}

void* performFilterParallel(
    void* arg
) 
{ 
    PerformFilterParameters_t *args = (PerformFilterParameters_t *)arg;

    cv::Mat *input = args->input;
    cv::Mat *result = args->result;
    cv::Mat *filter = args->filter;
    int rows = args->rows;
    int cols = args->cols;
    
    float pixelResult = 0.0;

    int mm, nn, ii, jj;

    for (int ik = 0; ik < 3; ++ik)
    {
        mm = 2 - ik;

        for (int jk = 0; jk < 3; ++jk)
        {
            nn = 2 - jk;
            
            ii = rows + (1 - mm);
            jj = cols + (1 - nn);

            if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
            {
                pixelResult += input->at<uchar>(ii, jj) * filter->at<float>(mm, nn);
            }

        }
    }

    result->at<uchar>(rows, cols) = cv::saturate_cast<uchar>(pixelResult);

    pthread_exit(NULL); 
} 
  
cv::Mat performParallelMethod(
    cv::Mat *currentFrame,
    Filters_t *filters
) 
{ 
    pthread_t thread; 

    int rows = currentFrame->rows;
    int cols = currentFrame->cols;
    int numThreads = rows * cols;

    std::vector<pthread_t> threads(numThreads);
    std::vector<PerformFilterParameters_t> args(numThreads);
    
    cv::Mat h1Results = cv::Mat::zeros(currentFrame->size(), currentFrame->type());
    printf("---h1---\n");

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            PerformFilterParameters_t performFiltersParameters_H1;
            initializePerformFilterParametersValues(
                &performFiltersParameters_H1, 
                currentFrame, 
                &h1Results, 
                &filters->h1, 
                i, 
                j
            );
            args.at(i * cols + j) = performFiltersParameters_H1;
            pthread_create(&thread, NULL, &performFilterParallel, &args.at(i * cols + j)); 
        }
    }

    for (int i = 0; i < numThreads; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    printf("Depois join\n");

    // threads.clear();
    // args.clear();

    cv::Mat h2Results = cv::Mat::zeros(currentFrame->size(), currentFrame->type());

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            PerformFilterParameters_t performFiltersParameters_H2;
            initializePerformFilterParametersValues(
                &performFiltersParameters_H2, 
                &h1Results, 
                &h2Results, 
                &filters->h2, 
                rows, 
                cols
            );
            args.at(i * cols + j) = performFiltersParameters_H2;
            pthread_create(&thread, NULL, &performFilterParallel, &args.at(i * cols + j)); 
        }
    }

    for (int i = 0; i < numThreads; ++i)
    {
        pthread_join(threads[i], NULL);
    }
  
    return h1Results;
} 


void playVideo(
    cv::VideoCapture *videoCapture,
    Filters_t *filters,
    const std::string video_path,
    const Method method
)
{
    cv::Mat currentFrame;
    cv::Mat performedFrame;

    int number_of_images = 0;
    float amount_time = 0.0;

    videoCapture->open(video_path);

    if (!videoCapture->isOpened())
    {
        std::cout << "Failed to open video: " << video_path << std::endl;
        return;
    }

    do
    {
        if (!videoCapture->read(currentFrame))
            break;

        cv::cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY); 

        clock_t begin_time = clock();

        switch (method)
        {
        case PARALLEL:
            performedFrame = performParallelMethod(&currentFrame, filters);
            break;
        case PARALLEL_OPEN_MP:
            /* code */
            break;
        default:
            performedFrame = performSequencialMethod(&currentFrame, filters);
            break;
        }

        number_of_images++;
        amount_time += float(clock() - begin_time ) / CLOCKS_PER_SEC;

        cv::imshow("Frame", performedFrame);
        cv::waitKey(1);

    } while (videoCapture->isOpened());

    std::string method_string;
    switch (method)
    {
    case PARALLEL:
        method_string = "Parallel";
        break;
    case PARALLEL_OPEN_MP:
        method_string = "Parallel OpenMP";
        break;
    default:
        method_string = "Sequencial";
        break;
    }
    std::cout << "Mean[" << method_string << "] = "<< amount_time << "/" << number_of_images << " = " << amount_time / number_of_images << std::endl;
}


int main()
{
    cv::VideoCapture videoCapture;
    Filters_t filters;

    playVideo(&videoCapture, &filters, "videos/1.mp4", SEQUENTIAL);
    //playVideo(&videoCapture, &filters, "videos/1.mp4", PARALLEL);
    //playVideo(&videoCapture, &filters, "videos/1.mp4", PARALLEL_OPEN_MP);

    return 0;
}