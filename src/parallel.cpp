#include <parallel.h>

#include <util.h>

void *h1ThreadFunction(
    void *arg
)
{
    PerformH1ParallelParameters_t *args = (PerformH1ParallelParameters_t *)arg;

    // Get parameters
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
        // Lock index to update it and copy it's value in local_index
        pthread_mutex_lock(mutex_index);
        
        if (*index == input->size())
        {
            pthread_mutex_unlock(mutex_index);
            break;
        }

        local_index = *index;
        (*index)++;
        
        pthread_mutex_unlock(mutex_index);

        // Initialize a matrix of zeros
        cv::Mat resultValue = cv::Mat::zeros(
            input->at(local_index).size(), 
            input->at(local_index).type()
        );

        // Perform filter
        performFilter(
            &input->at(local_index),
            &resultValue,
            filter,
            input->at(local_index).size().height,
            input->at(local_index).size().width
        );

        // Store result
        result->at(local_index) = resultValue;

        // Lock the index_to_process queue to show new index to perform h2 in the second block of threads.
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

    // Get parameters
    std::vector<cv::Mat> *input = args->input;
    std::vector<cv::Mat> *result = args->result;
    cv::Mat *filter = args->filter;
    std::queue<int> *indexes_to_process = args->indexes_to_process;
    pthread_mutex_t *mutex_indexes_to_process = args->mutex_indexes_to_process;
    int *counter_frames_processed = args->counter_frames_processed;

    int local_index;
    while (true)
    {
        // Lock indexes_to_process to analyse if there is index to process
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

        // If there is at least one index to perform h2, the following logic is applied
        if (local_index != -1)
        {
            // Initialize a matrix of zeros
            cv::Mat resultValue = cv::Mat::zeros(
                input->at(local_index).size(), 
                input->at(local_index).type()
            );

            // Perform filter
            performFilter(
                &input->at(local_index),
                &resultValue,
                filter,
                input->at(local_index).size().height,
                input->at(local_index).size().width
            );

            // Store result
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
    // The goal is to perform h1 filter in parallel in all frames from all_video_frames array and store it in a buffer.
    // The buffer is consumed by another set of threads to perform h2 in parallel too.
    // The result is stored in the all_performed_video_frames.

    std::cout << "Performing parallel method ... " << std::endl;

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

    // Define parameters and initiliaze threads to perform h1
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

    // Define parameters and initiliaze threads to perform h2
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

    // Join h1 threads
    for (int i = 0; i < *numThreads; ++i) 
    {
        pthread_join(threads_input[i], NULL);
    }

    // Join h2 threads
    for (int i = 0; i < *numThreads; ++i) 
    {
        pthread_join(threads_output[i], NULL);
    }

    // Destroy mutex
    pthread_mutex_destroy(&mutex_index_to_process_input);
    pthread_mutex_destroy(&mutex_index_to_process_output);
}