#include <unordered_map>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class HistogramContrastSaliency {
   public:
    // A map from a color 3-tuple to T.
    template <class T>
    using ColorMap =
        unordered_map<int, unordered_map<int, unordered_map<int, T>>>;
    // src - The input BGR image.
    HistogramContrastSaliency(const Mat& src);
    // Computes the saliency map using the Histogram Contrast based saliency
    // method.
    Mat GetSaliencyMap();

   private:
    // Quantizes each of the channels in the input image into nBins.
    // src       - The input image.
    // nBins     - Number of bins to quantize each channel.
    // quantized - The quantized output image.
    void Quantize(const int nBins, Mat* quantized);
    // Converts a BGR color to CIE-Lab space.
    Vec3i BGRToLab(const Vec3b& bgr);
    // Stores reference to the input image.
    const Mat& src_;
};
