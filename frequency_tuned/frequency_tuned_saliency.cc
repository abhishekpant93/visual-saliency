#include "frequency_tuned_saliency.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

FrequencyTunedSaliency::FrequencyTunedSaliency(const Mat &src) : src_(src) {}

// Frequency tuned salient region detection.
Mat FrequencyTunedSaliency::GetSaliencyMap() {
    if (src_.channels() != 3) return Mat();

    // Convolve the src_ image with a symmetric 5x5 Gaussian kernel.
    Mat blur;
    GaussianBlur(src_, blur, Size(5, 5), 0);

    // Convert the blurred image to CIE Lab color space.
    cvtColor(blur, blur, CV_BGR2Lab);

    // Get the mean pixel value for each channel in the Lab space.
    const Scalar mean_sc = mean(blur);
    const Vec3b mean_px{static_cast<uchar>(mean_sc[0]),
                        static_cast<uchar>(mean_sc[1]),
                        static_cast<uchar>(mean_sc[2])};

    // Compute the saliency map.
    Mat output(src_.size(), CV_8UC1);
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            const int val = norm(mean_px, blur.at<Vec3b>(y, x));
            output.at<uchar>(y, x) = static_cast<uchar>(val);
        }
    }
    return output;
}
