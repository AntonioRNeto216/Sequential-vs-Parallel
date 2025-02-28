#include <iostream>
#include <chrono>

#include <util.h>
#include <defines.h>
#include <parallel.h>
#include <sequential.h>
#include <parallel_open_mp.h>


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

    // Get all frames from a video
    getFrames(&all_video_frames, &videoCapture, &video);

    // Array to store all performed frames
    std::vector<cv::Mat> all_performed_video_frames(all_video_frames.size());

    // Start time
    auto begin_time = std::chrono::high_resolution_clock::now();

    // Call function to perform the chosen method
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

    // End time
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Time to perform filters = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << "ms" << std::endl;

    // Play video to show the result to user
    playPerformedVideo(&all_performed_video_frames);
    
    return 0;
}