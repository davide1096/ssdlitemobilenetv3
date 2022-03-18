#include <math.h>
#include <algorithm>
#include <vector>
#include <platform.h>
#include <ssdlitembnv3.h>


template<class T>
const T& clamp(const T& v, const T& lo, const T& hi)
{
    assert(!(hi < lo));
    return v < lo ? lo : hi < v ? hi : v;
}

//define VOC labels
static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };

ssdlitemobilenetv3::ssdlitemobilenetv3(){
    numThreads = 4;
    target_size = 300;
    thresh = 0.4;
    //float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    //float norm_vals[3] = {1.0f, 1.0f, 1.0f};
}

ssdlitemobilenetv3::~ssdlitemobilenetv3()
{
    ;
}

int ssdlitemobilenetv3::init(const bool use_vulkan_compute){
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_int8_inference = true;
    net.opt.use_vulkan_compute = use_vulkan_compute;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.use_int8_storage = true;
    net.opt.use_int8_arithmetic = true;
    net.opt.use_packing_layout = true;
    net.opt.use_shader_pack8 = false;
    net.opt.use_image_storage = false;

    net.opt.use_bf16_storage = true;

    return 0;
}

int ssdlitemobilenetv3::loadModel(const char* paramPath, const char* binPath){
    net.load_param(paramPath);
    net.load_model(binPath);

    printf("Ncnn model init success!\n");

    return 0;
}

int ssdlitemobilenetv3::inference(const cv::Mat& srcImg, std::vector<Object>& objects){
    //resizing of input image data
    int img_w = srcImg.cols;
    int img_h = srcImg.rows;
    ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                       img_w, img_h, target_size, target_size);
    //Normalization of input image data
    inputImg.substract_mean_normalize(mean_vals, norm_vals);

    //extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(4);

    ex.input("input", inputImg);

    ncnn::Mat out;
    ex.extract("detection_out", out);

    objects.clear();

    //std::cout<<out.w<<out.h;  This outputs 6 and 5, which means out is a matrix 5x6
    // I guess its 5 boxes detected in this image with 6 elements each

    for (int i=0; i<out.h; i++){
        const float* values = out.row(i);
        Object object;
        object.label = values[0];
        object.prob = values[1];
        if (object.prob>thresh){
            float x1 = clamp(values[2] * target_size, 0.f, float(target_size - 1)) / target_size * img_w;
            float y1 = clamp(values[3] * target_size, 0.f, float(target_size - 1)) / target_size * img_h;
            float x2 = clamp(values[4] * target_size, 0.f, float(target_size - 1)) / target_size * img_w;
            float y2 = clamp(values[5] * target_size, 0.f, float(target_size - 1)) / target_size * img_h;

            object.rect.x = x1;
            object.rect.y = y1;
            object.rect.width = x2 - x1;
            object.rect.height = y2 - y1;

            objects.push_back(object);
        }
    }

    return 0;

}

int ssdlitemobilenetv3::drawObjects(const cv::Mat& image, const std::vector<Object>& objects){
    //cv::Mat image = bgr.clone();
    for (size_t i=0; i <objects.size(); i++){
        //if (objects[i].prob>thresh){
            const Object& obj = objects[i];

            cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob*100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y<0) y=0;
            if (x + label_size.width > image.cols) x=image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 0));
        //}
     }

     cv::imshow("image", image);
     return 0;
}
