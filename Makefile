CFLAGS := -O3 -std=c++11 -Wall $(CFLAGS)

OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`

HISTOGRAM_CONTRAST = histogram_contrast/histogram_contrast_saliency.o
FREQUENCY_TUNED = frequency_tuned/frequency_tuned_saliency.o

SALIENCY_METHODS = $(HISTOGRAM_CONTRAST) $(FREQUENCY_TUNED)

all: $(SALIENCY_METHODS) main.cc
	clang++ -g $(OPENCV_CFLAGS) $(CFLAGS) -o saliency main.cc $(SALIENCY_METHODS) $(OPENCV_LIBS) $(LDFLAGS)

$(HISTOGRAM_CONTRAST): $(wildcard histogram_contrast/*.cc) $(wildcard histogram_contrast/*.hpp)
	cd histogram_contrast && make all

$(FREQUENCY_TUNED): $(wildcard frequency_tuned/*.cc) $(wildcard frequency_tuned/*.hpp)
	cd frequency_tuned && make all

clean:
	\rm *.o *~
