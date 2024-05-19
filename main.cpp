#include <iostream>
#include <queue>
#include <vector>
#include <unistd.h> 
#include <pthread.h>
#include <opencv2/opencv.hpp>


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
}

void performSequencialMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters
)
{
    cv::Mat currentFrame;
    for (int i = 0; i < all_video_frames->size(); ++i)
    {
        currentFrame = all_video_frames->at(i);

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

        all_performed_video_frames->at(i) = h2Results;
    }
}

void *h1ThreadFunction(
    void *arg
)
{
    PerformH1ParallelParameters_t *args = (PerformH1ParallelParameters_t *)arg;

    std::vector<cv::Mat> *input = args->input;
    std::vector<cv::Mat> *result = args->result;
    cv::Mat *filter = args->filter;
    int *index = args->index;
    pthread_mutex_t *mutex_index = args->mutex_index;
    std::queue<int> *indexes_to_process = args->indexes_to_process;
    pthread_mutex_t *mutex_indexes_to_process = args->mutex_indexes_to_process;

    int local_index;
    while (true)
    {
        pthread_mutex_lock(mutex_index);
        
        if (*index == input->size())
        {
            pthread_mutex_unlock(mutex_index);
            break;
        }

        local_index = *index;
        (*index)++;
        
        pthread_mutex_unlock(mutex_index);

        cv::Mat resultValue = cv::Mat::zeros(
            input->at(local_index).size(), 
            input->at(local_index).type()
        );

        performFilter(
            &input->at(local_index),
            &resultValue,
            filter,
            input->at(local_index).size().height,
            input->at(local_index).size().width
        );

        result->at(local_index) = resultValue;

        pthread_mutex_lock(mutex_indexes_to_process);
        indexes_to_process->push(local_index);
        pthread_mutex_unlock(mutex_indexes_to_process);
    }

    pthread_exit(NULL);
}

void *h2ThreadFunction(
    void *arg
)
{
    PerformH2ParallelParameters_t *args = (PerformH2ParallelParameters_t *)arg;

    std::vector<cv::Mat> *input = args->input;
    std::vector<cv::Mat> *result = args->result;
    cv::Mat *filter = args->filter;
    std::queue<int> *indexes_to_process = args->indexes_to_process;
    pthread_mutex_t *mutex_indexes_to_process = args->mutex_indexes_to_process;
    int *counter_frames_processed = args->counter_frames_processed;

    int local_index;
    while (true)
    {
        pthread_mutex_lock(mutex_indexes_to_process);
        
        if (indexes_to_process->size() > 0)
        {
            local_index = indexes_to_process->front();
            indexes_to_process->pop();
            (*counter_frames_processed)++;
        }
        else if (*counter_frames_processed == input->size())
        {
            pthread_mutex_unlock(mutex_indexes_to_process);
            break;
        }
        else
        {
            local_index = -1;
        }
        
        pthread_mutex_unlock(mutex_indexes_to_process);

        if (local_index != -1)
        {
            cv::Mat resultValue = cv::Mat::zeros(
                input->at(local_index).size(), 
                input->at(local_index).type()
            );

            performFilter(
                &input->at(local_index),
                &resultValue,
                filter,
                input->at(local_index).size().height,
                input->at(local_index).size().width
            );

            result->at(local_index) = resultValue;
        }
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
    std::vector<PerformH1ParallelParameters_t> vector_performH1ParallelParameters(*numThreads);

    pthread_mutex_t mutex_index_to_process_input;
    pthread_mutex_init(&mutex_index_to_process_input, NULL);
    
    // Buffer -> output
    int counter_frames_processed = 0;
    std::queue<int> indexes_to_process_buffer;
    std::vector<pthread_t> threads_output(*numThreads);
    std::vector<PerformH2ParallelParameters_t> vector_performH2ParallelParameters(*numThreads);

    pthread_mutex_t mutex_index_to_process_output;
    pthread_mutex_init(&mutex_index_to_process_output, NULL);

    for (int i = 0; i < *numThreads; ++i)
    {
        vector_performH1ParallelParameters[i].input = all_video_frames;
        vector_performH1ParallelParameters[i].result = &buffer;
        vector_performH1ParallelParameters[i].filter = &filters->h1;
        vector_performH1ParallelParameters[i].index = &index_to_process_input;
        vector_performH1ParallelParameters[i].mutex_index = &mutex_index_to_process_input;
        vector_performH1ParallelParameters[i].indexes_to_process = &indexes_to_process_buffer;
        vector_performH1ParallelParameters[i].mutex_indexes_to_process = &mutex_index_to_process_output;

        pthread_create(&threads_input[i], NULL, h1ThreadFunction, (void*)&vector_performH1ParallelParameters[i]);
    }

    for (int i = 0; i < *numThreads; ++i)
    {
        vector_performH2ParallelParameters[i].input = &buffer;
        vector_performH2ParallelParameters[i].result = all_performed_video_frames;
        vector_performH2ParallelParameters[i].filter = &filters->h2;
        vector_performH2ParallelParameters[i].indexes_to_process = &indexes_to_process_buffer;
        vector_performH2ParallelParameters[i].mutex_indexes_to_process = &mutex_index_to_process_output;
        vector_performH2ParallelParameters[i].counter_frames_processed = &counter_frames_processed;

        pthread_create(&threads_output[i], NULL, h2ThreadFunction, (void*)&vector_performH2ParallelParameters[i]);
    }

    for (int i = 0; i < *numThreads; ++i) 
    {
        pthread_join(threads_input[i], NULL);
    }

    for (int i = 0; i < *numThreads; ++i) 
    {
        pthread_join(threads_output[i], NULL);
    }

    pthread_mutex_destroy(&mutex_index_to_process_input);
    pthread_mutex_destroy(&mutex_index_to_process_output);
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

    clock_t begin_time;
    clock_t end_time;

    std::cout << "Getting source video ... " << std::endl;
    getFrames(&all_video_frames, &videoCapture, &video);

    std::vector<cv::Mat> all_performed_video_frames(all_video_frames.size());

    begin_time = clock();

    std::cout << "Performing filters ... " << std::endl;
    //performSequencialMethod(&all_video_frames, &all_performed_video_frames, &filters);
    performParallelMethod(&all_video_frames, &all_performed_video_frames, &filters, &numThreads);

    end_time = clock();

    std::cout << "Time to perform filters = " << (end_time - begin_time) / CLOCKS_PER_SEC << "s" << std::endl;

    std::cout << "Playing performed video ... " << std::endl;
    playPerformedVideo(&all_performed_video_frames);

    return 0;
}