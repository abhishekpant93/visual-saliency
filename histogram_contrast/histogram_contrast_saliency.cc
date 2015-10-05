#include "histogram_contrast_saliency.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

HistogramContrastSaliency::HistogramContrastSaliency(const Mat &src)
    : src_(src) {}

Mat HistogramContrastSaliency::GetSaliencyMap() {
    if (src_.channels() != 3) return Mat();

    // Quantize each of the input channels.
    int nBins = 12;
    Mat quantized_bgr;
    Quantize(nBins, &quantized_bgr);

    // A map storing the counts of the various BGR colors present in the
    // quantized image.
    HistogramContrastSaliency::ColorMap<int> histogram;
    for (int y = 0; y < quantized_bgr.rows; ++y) {
        for (int x = 0; x < quantized_bgr.cols; ++x) {
            Vec3b px = quantized_bgr.at<Vec3b>(y, x);
            histogram[px[0]][px[1]][px[2]] += 1;
        }
    }

    // A map storing the saliency value for each BGR color in the image.
    // This method does not take advantage of spatial relationships, so
    // saliency value is the same for all pixels of a color.
    HistogramContrastSaliency::ColorMap<float> saliency_bgr;
    for (const auto &iter_b : histogram) {
        for (const auto &iter_g : iter_b.second) {
            for (const auto &iter_r : iter_g.second) {
                uchar b = iter_b.first;
                uchar g = iter_g.first;
                uchar r = iter_r.first;
                const Vec3i lab = BGRToLab(Vec3b{b, g, r});

                // Skip if already calculated.
                if (saliency_bgr[b][g][r] > 0) continue;

                // Calculate the saliency value for (b, g, r).
                for (const auto &iter_bb : histogram) {
                    for (const auto &iter_gg : iter_bb.second) {
                        for (const auto &iter_rr : iter_gg.second) {
                            uchar bb = iter_bb.first;
                            uchar gg = iter_gg.first;
                            uchar rr = iter_rr.first;
                            const Vec3i llab = BGRToLab(Vec3b{bb, gg, rr});

                            // Probability of this color (bb, gg, rr) occurring
                            // in the image.
                            float prob =
                                ((float)(histogram[bb][gg][rr])) /
                                (quantized_bgr.rows * quantized_bgr.cols);
                            // Euclidean distance between the two colors in the
                            // Lab color space, for perceptual proximity.
                            float D = norm(lab, llab);
                            saliency_bgr[b][g][r] += prob * D;
                        }
                    }
                }
            }
        }
    }

    // Fill in the output image using the computed saliency values.
    Mat output(src_.size(), CV_32F);
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            Vec3b px = quantized_bgr.at<Vec3b>(y, x);
            output.at<float>(y, x) = saliency_bgr[px[0]][px[1]][px[2]];
        }
    }
    normalize(output, output, 0, 1, NORM_MINMAX);

    return output;
}

void HistogramContrastSaliency::Quantize(const int nBins, Mat *quantized) {
    if (quantized == nullptr || src_.channels() != 3 || nBins < 1 ||
        nBins >= 255)
        return;

    int binSz = 256 / nBins;
    *quantized = Mat(src_.size(), src_.type());
    for (int y = 0; y < src_.rows; ++y) {
        for (int x = 0; x < src_.cols; ++x) {
            const Vec3b src__px = src_.at<Vec3b>(y, x);
            const int b = src__px[0], g = src__px[1], r = src__px[2];
            Vec3b &q_px = quantized->at<Vec3b>(y, x);
            q_px[0] = (b / binSz == nBins ? nBins - 1 : b / binSz) * binSz;
            q_px[1] = (g / binSz == nBins ? nBins - 1 : g / binSz) * binSz;
            q_px[2] = (r / binSz == nBins ? nBins - 1 : r / binSz) * binSz;
        }
    }
}

Vec3i HistogramContrastSaliency::BGRToLab(const Vec3b &bgr) {
    Mat m(1, 1, CV_8UC3);
    m.at<Vec3b>(0, 0) = bgr;
    cvtColor(m, m, CV_BGR2Lab);
    Vec3i lab = m.at<Vec3b>(0, 0);
    // Undo the OpenCV scaling, so that Euclidean proximity in Lab space is
    // indicative of perceptual similarity.
    lab[0] *= 100.0 / 255;
    lab[1] -= 128;
    lab[2] -= 128;
    return lab;
}
