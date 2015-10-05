CFLAGS := -O3 -std=c++11 -Wall $(CFLAGS)
OPENCV_CFLAGS = `pkg-config --cflags opencv`

all: frequency_tuned_saliency.hpp frequency_tuned_saliency.cc
	clang++ -g $(OPENCV_CFLAGS) $(CFLAGS) -c frequency_tuned_saliency.cc

clean:
	\rm *.o *~
