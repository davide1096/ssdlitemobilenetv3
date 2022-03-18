#ifndef SSDLITEMBNV3_H_INCLUDED
#define SSDLITEMBNV3_H_INCLUDED

#include "net.h"
#include <vector>
#include <opencv2/opencv.hpp>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;

};

class ssdlitemobilenetv3{
private:
    ncnn::Net net;
    int numThreads;
    int target_size;
    float thresh;
    float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    float norm_vals[3] = {1.0f, 1.0f, 1.0f};

public:
    ssdlitemobilenetv3();
    ~ssdlitemobilenetv3();
    int init(const bool use_vulkan_compute=false);
    int loadModel(const char* paramPath, const char* binPath);
    int inference(const cv::Mat& srcImg, std::vector<Object>& objects);
    int drawObjects(const cv::Mat& image, const std::vector<Object>& objects);
};


#endif // SSDLITEMBNV3_H_INCLUDED
