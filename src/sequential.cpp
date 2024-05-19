#include <sequential.h>

#include <util.h>
#include <defines.h>

void performSequencialMethod(
    std::vector<cv::Mat> *all_video_frames,
    std::vector<cv::Mat> *all_performed_video_frames,
    Filters_t *filters
)
{
    std::cout << "Performing sequencial method ... " << std::endl;

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