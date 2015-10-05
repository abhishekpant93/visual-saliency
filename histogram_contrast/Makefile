CFLAGS := -O3 -std=c++11 -Wall $(CFLAGS)
OPENCV_CFLAGS = `pkg-config --cflags opencv`

all: histogram_contrast_saliency.hpp histogram_contrast_saliency.cc
	clang++ -g $(OPENCV_CFLAGS) $(CFLAGS) -c histogram_contrast_saliency.cc

clean:
	\rm *.o *~
