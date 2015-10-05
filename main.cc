#include <iostream>

#include "frequency_tuned/frequency_tuned_saliency.hpp"
#include "histogram_contrast/histogram_contrast_saliency.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage: ./saliency <path_to_image>" << endl;
        return 0;
    }

    // Read the image.
    const Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    imshow("src", src);

    // Compute saliency using Histogram Contrast method.
    HistogramContrastSaliency hcs(src);
    Mat hcs_output = hcs.GetSaliencyMap();
    imshow("histogram contrast saliency", hcs_output);

    // Compute saliency using Frequency Tuned method.
    FrequencyTunedSaliency fts(src);
    Mat fts_output = fts.GetSaliencyMap();
    imshow("frequency tuned saliency", fts_output);

    waitKey(0);

    return 0;
}
