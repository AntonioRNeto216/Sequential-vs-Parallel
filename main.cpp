#include <iostream>
#include <chrono>

#include <util.h>
#include <defines.h>
#include <parallel.h>
#include <sequential.h>


#include <omp.h>

void performFilterOpenMP(
    cv::Mat *input,
    cv::Mat *result,
    cv::Mat *filter,
    int rows,
    int cols
)
{
    int i, j, ik, jk, mm, nn, ii, jj;
    float pixelResult;

    #pragma omp parallel private(i, j, ik, jk, mm, nn, ii, jj, pixelResult)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < rows; ++i)
        {
            for (j = 0; j < cols; ++j)
            {
                pixelResult = 0.0;

                for (ik = 0; ik < 3; ++ik)
                {
                    mm = 2 - ik;

                    for (jk = 0; jk < 3; ++jk)
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
}

void performParallelOpenMPMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters,
    int *numThreads
) 
{
    std::cout << "Performing parallel with OpenMP method ... " << std::endl;

    std::vector<cv::Mat> buffer;

    omp_set_num_threads(*numThreads);

    for (int i = 0; i < all_video_frames->size(); ++i)
    {

        cv::Mat h1Results = cv::Mat::zeros(
            all_video_frames->at(i).size(), 
            all_video_frames->at(i).type()
        );

        cv::Mat h2Results = cv::Mat::zeros(
            all_video_frames->at(i).size(), 
            all_video_frames->at(i).type()
        );

        performFilterOpenMP(
            &all_video_frames->at(i),
            &h1Results,
            &filters->h1,
            all_video_frames->at(i).size().height,
            all_video_frames->at(i).size().width
        );

        performFilterOpenMP(
            &h1Results,
            &h2Results,
            &filters->h2,
            all_video_frames->at(i).size().height,
            all_video_frames->at(i).size().width
        );

        all_performed_video_frames->at(i) = h2Results;

    }
}

int main()
{
    // Initilize some variables
    Filters_t filters;
    cv::VideoCapture videoCapture;

    std::vector<cv::Mat> all_video_frames;

    std::string video;
    Method method;
    int numThreads;

    // Menus
    switch (videoMenu())
    {
        case 1:
            video = "videos/1.mp4";
            break;
        case 2:
            video = "videos/2.mp4";
            break;
        case 3:
            video = "videos/3.mp4";
            break;
        default:
            video = "videos/4.mp4";
            break;
    }

    method = methodMenu();

    if (method != SEQUENTIAL)
    {
        numThreads = numThreadsMenu();
    }
    else
    {
        numThreads = 0;
    }

    // Logic process
    getFrames(&all_video_frames, &videoCapture, &video);

    std::vector<cv::Mat> all_performed_video_frames(all_video_frames.size());

    auto begin_time = std::chrono::high_resolution_clock::now();

    switch (method)
    {
        case SEQUENTIAL:
            performSequencialMethod(&all_video_frames, &all_performed_video_frames, &filters);
            break;
        case PARALLEL:
            performParallelMethod(&all_video_frames, &all_performed_video_frames, &filters, &numThreads);
            break;
        default:
            performParallelOpenMPMethod(&all_video_frames, &all_performed_video_frames, &filters, &numThreads);
            break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Time to perform filters = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << "ms" << std::endl;

    playPerformedVideo(&all_performed_video_frames);
    
    return 0;
}