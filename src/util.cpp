#include <util.h>

void performFilter(
    cv::Mat *input,
    cv::Mat *result,
    cv::Mat *filter,
    int rows,
    int cols
)
{
    // Perform convolution operation on the input frame with filter parameter.
    // The result is stored in result parameter

    int mm, nn, ii, jj;
    float pixelResult;

    // Iterate over all pixels performing a kernel with size 3x3 with padding
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

void getFrames(
    std::vector<cv::Mat> *all_video_frames,
    cv::VideoCapture *videoCapture,
    std::string *video_path
)
{
    // Get all frames from a source video

    std::cout << std::endl << "Getting source video ... " << std::endl;

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
    // Playing performed video to user the process result
    
    std::cout << "Playing performed video ... " << std::endl;
    
    const std::string windowName = "Perfomed Video";

    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    for (int i = 0; i < all_video_frames->size(); ++i)
    {
        cv::imshow(windowName, all_video_frames->at(i));
        cv::waitKey(33);
    }

    cv::destroyWindow(windowName);
}

int videoMenu()
{
    // User can choose video 1, 2, 3  or 4

    int input;

    std::cout << std::endl << "Choose a video" << std::endl;
    std::cout << "1) videos/1.mp4 " << std::endl;
    std::cout << "2) videos/2.mp4 " << std::endl;
    std::cout << "3) videos/3.mp4 " << std::endl;
    std::cout << "4) videos/4.mp4 " << std::endl;
    std::cout << "Any) Quit " << std::endl;
    std::cout << "Input: ";
    std::cin >> input;

    if (input == 1 || input == 2 || input == 3 || input == 4)
    {
        return input;
    }
    else
    {
        exit(0);
    }
}

Method methodMenu()
{
    // User can choose a method (Sequential, Parallel - Pthreads or Parallel - OpenMP)

    int input;

    std::cout << std::endl << "Choose a method" << std::endl;
    std::cout << "1) Sequential " << std::endl;
    std::cout << "2) Parallel (Pthreds)" << std::endl;
    std::cout << "3) Parallel (OpenMP) " << std::endl;
    std::cout << "Any) Quit " << std::endl;
    std::cout << "Input: ";
    std::cin >> input;

    if (input == 1)
    {
        return SEQUENTIAL;
    }
    else if (input == 2)
    {
        return PARALLEL;
    }
    else if (input == 3)
    {
        return PARALLEL_OPEN_MP;
    }
    else
    {
        exit(0);
    }
}

int numThreadsMenu()
{
    // User can choose the number of threads

    int numThreads;

    std::cout << "Choose number of threads: ";
    std::cin >> numThreads;

    return numThreads;
}