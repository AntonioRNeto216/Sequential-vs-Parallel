#include <iostream>

#include <util.h>
#include <defines.h>
#include <parallel.h>
#include <sequential.h>


int main()
{
    // Initilize some variables
    Filters_t filters;
    cv::VideoCapture videoCapture;

    std::vector<cv::Mat> all_video_frames;

    clock_t begin_time;
    clock_t end_time;

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

    begin_time = clock();

    switch (method)
    {
        case SEQUENTIAL:
            performSequencialMethod(&all_video_frames, &all_performed_video_frames, &filters);
            break;
        case PARALLEL:
            performParallelMethod(&all_video_frames, &all_performed_video_frames, &filters, &numThreads);
            break;
        default:

            break;
    }

    end_time = clock();

    std::cout << "Time to perform filters = " << (end_time - begin_time) / CLOCKS_PER_SEC << "s" << std::endl;

    playPerformedVideo(&all_performed_video_frames);
    
    return 0;
}