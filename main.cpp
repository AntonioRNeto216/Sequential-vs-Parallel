#include <iostream>
#include <vector>
#include <unistd.h> 
#include <pthread.h>
#include <opencv2/opencv.hpp>


typedef struct Filters_s
{
    cv::Mat h1 = (cv::Mat_<float>(3,3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    cv::Mat h2 = (cv::Mat_<float>(3,3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
} Filters_t;

typedef struct PerformFilterParallelParameters_s
{
    std::vector<cv::Mat> *input;
    std::vector<cv::Mat> *result;
    cv::Mat *filter;
    int *index;
    pthread_mutex_t *mutex;
} PerformFilterParallelParameters_t;

void performFilter(
    cv::Mat *input,
    cv::Mat *result,
    cv::Mat *filter,
    int rows,
    int cols
)
{
    int mm, nn, ii, jj;
    float pixelResult;

    printf("num rows: %d| num cols: %d \n", rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
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

                    if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
                    {
                        pixelResult += input->at<uchar>(ii, jj) * filter->at<float>(mm, nn);
                    }

                }
            }

            result->at<uchar>(i, j) = cv::saturate_cast<uchar>(pixelResult);
        }
    }

    printf("Acabou\n");
}

void performSequencialMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters
)
{
    for (cv::Mat currentFrame : *all_video_frames)
    {
        int rows = currentFrame.rows;
        int cols = currentFrame.cols;
        
        // Perform h1 filter

        cv::Mat h1Results = cv::Mat::zeros(
            currentFrame.size(), 
            currentFrame.type()
        );

        cv::Mat h2Results = cv::Mat::zeros(
            currentFrame.size(), 
            currentFrame.type()
        );
        
        performFilter(
            &currentFrame, 
            &h1Results, 
            &filters->h1, 
            rows, 
            cols
        );

        // Perform h2 filter

        performFilter(
            &h1Results, 
            &h2Results, 
            &filters->h2, 
            rows, 
            cols
        );

        all_performed_video_frames->push_back(h2Results);
    }
}

void *threadFunction(
    void *arg
)
{
    PerformFilterParallelParameters_t *args = (PerformFilterParallelParameters_t *)arg;

    std::vector<cv::Mat> *input = args->input;
    std::vector<cv::Mat> *result = args->result;
    cv::Mat *filter = args->filter;
    int *index = args->index;
    pthread_mutex_t *mutex = args->mutex;

    int local_index;
    while (true)
    {
        pthread_mutex_lock(mutex);
        
        if (*index == input->size() - 1)
        {
            printf("Dentro IF\n");
            pthread_mutex_unlock(mutex);
            break;
        }

        printf("index %d\n", *index);
        local_index = *index;
        printf("index %d\n", *index);
        (*index)++;
        
        pthread_mutex_unlock(mutex);

        // performFilter(
        //     &input->at(local_index),
        //     &result->at(local_index),
        //     filter,
        //     input->at(local_index).size().height,
        //     input->at(local_index).size().width
        // );

        //printf("Depois filtro (index = %d)\n", *index);
    }

    pthread_exit(NULL);
}

void performParallelMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters,
    int *numThreads
)
{
    // Input -> buffer
    int index_to_process_input = 0;
    std::vector<pthread_t> threads_input(*numThreads);
    std::vector<cv::Mat> buffer(all_video_frames->size());
    std::vector<PerformFilterParallelParameters_t> vector_performFilterParallelParameters_input;

    pthread_mutex_t mutex_index_to_process_input;
    pthread_mutex_init(&mutex_index_to_process_input, NULL);
    
    // Buffer -> output
    // int index_to_process_buffer = -1;
    // std::vector<pthread_t> threads_output(*numThreads);
    // std::vector<PerformFilterParallelParameters_t> vector_performFilterParallelParameters_buffer;

    // pthread_mutex_t mutex_index_to_process_output;
    // pthread_mutex_init(&mutex_index_to_process_output, NULL);

    for (int i = 0; i < *numThreads; ++i)
    {
        PerformFilterParallelParameters_t performFilterParallelParameters_input;
        performFilterParallelParameters_input.input = all_video_frames;
        performFilterParallelParameters_input.result = &buffer;
        performFilterParallelParameters_input.filter = &filters->h1;
        performFilterParallelParameters_input.index = &index_to_process_input;
        performFilterParallelParameters_input.mutex = &mutex_index_to_process_input;

        vector_performFilterParallelParameters_input.push_back(performFilterParallelParameters_input);

        pthread_create(&threads_input[i], NULL, threadFunction, (void*)&vector_performFilterParallelParameters_input[i]);

        // PerformFilterParallelParameters_t performFilterParallelParameters_buffer;
        // performFilterParallelParameters_buffer.input = &buffer;
        // performFilterParallelParameters_buffer.result = all_performed_video_frames;
        // performFilterParallelParameters_buffer.filter = &filters->h2;
        // performFilterParallelParameters_buffer.index = &index_to_process_buffer;
        // performFilterParallelParameters_buffer.mutex = &mutex_index_to_process_output;

        // vector_performFilterParallelParameters_buffer.push_back(performFilterParallelParameters_buffer);

        // pthread_create(&threads_output[i], NULL, threadFunction, (void*)&vector_performFilterParallelParameters_buffer[i]);
    }

    for (int i = 0; i < *numThreads; ++i) 
    {
        pthread_join(threads_input[i], NULL);
        // pthread_join(threads_output[i], NULL);
    }

    pthread_mutex_destroy(&mutex_index_to_process_input);
    // pthread_mutex_destroy(&mutex_index_to_process_output);
}

void getFrames(
    std::vector<cv::Mat> *all_video_frames,
    cv::VideoCapture *videoCapture,
    std::string *video_path
)
{
    videoCapture->open(*video_path);

    if (!videoCapture->isOpened())
    {
        std::cout << "Failed to open video: " << *video_path << std::endl;
        return;
    }

    cv::Mat currentFrame;
    
    while(videoCapture->read(currentFrame))
    {
        cv::cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY);
        all_video_frames->push_back(currentFrame.clone());
    }

    videoCapture->release();
}

void playPerformedVideo(
    std::vector<cv::Mat> *all_video_frames
)
{
    const std::string windowName = "Perfomed Video";

    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    for (int i = 0; i < all_video_frames->size(); ++i)
    {
        cv::imshow(windowName, all_video_frames->at(i));
        cv::waitKey(33);
    }

    cv::destroyWindow(windowName);
}

int main()
{
    // -------
    int numThreads = 4;
    std::string video = "videos/1.mp4";
    // -------

    Filters_t filters;
    cv::VideoCapture videoCapture;

    std::vector<cv::Mat> all_video_frames;
    std::vector<cv::Mat> all_performed_video_frames;

    clock_t begin_time;
    clock_t end_time;

    std::cout << "Getting source video ... " << std::endl;
    getFrames(&all_video_frames, &videoCapture, &video);

    begin_time = clock();

    std::cout << "Performing filters ... " << std::endl;
    performParallelMethod(&all_video_frames, &all_performed_video_frames, &filters, &numThreads);

    end_time = clock();

    std::cout << "Time to perform filters = " << (end_time - begin_time) / CLOCKS_PER_SEC << "s" << std::endl;

    std::cout << "Playing performed video ... " << std::endl;
    playPerformedVideo(&all_performed_video_frames);

    return 0;
}