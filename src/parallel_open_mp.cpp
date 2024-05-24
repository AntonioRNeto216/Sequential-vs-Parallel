#include <parallel_open_mp.h>


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

    #pragma omp parallel
    {
        #pragma omp single
        {
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

                #pragma omp task firstprivate(i, h1Results, h2Results)
                {

                    performFilterOpenMP(
                        &all_video_frames->at(i),
                        &h1Results,
                        &filters->h1,
                        all_video_frames->at(i).size().height,
                        all_video_frames->at(i).size().width
                    );

                    #pragma omp task firstprivate(i, h1Results, h2Results) depend(in: h1Results)
                    {

                        performFilterOpenMP(
                            &h1Results,
                            &h2Results,
                            &filters->h2,
                            all_video_frames->at(i).size().height,
                            all_video_frames->at(i).size().width
                        );

                        #pragma omp task firstprivate(i, h2Results) depend(out: h2Results)
                        {
                            all_performed_video_frames->at(i) = h2Results;
                        }
                    }
                }
            }
        }
    }
}