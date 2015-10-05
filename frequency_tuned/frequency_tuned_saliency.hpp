#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class FrequencyTunedSaliency {
   public:
    // src - Input BGR image.
    FrequencyTunedSaliency(const Mat& src);
    // Computes the saliency map from the input image using Frequency Tuned
    // Salient Region Detection technique.
    Mat GetSaliencyMap();

   private:
    // Stores a reference to the input image.
    const Mat& src_;
};
