#include <lccv.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ssdlitembnv3.h>

ssdlitemobilenetv3 mb3;

int main()
{
    cv::Mat frame;
    std::vector<Object> objects;

    mb3.init(false);
    mb3.loadModel("mobilenetv3_ssdlite_voc.param","mobilenetv3_ssdlite_voc.bin");

    //std::cout<<"Sample program for LCCV video capture"<<std::endl;
    std::cout<<"Press ESC to stop."<<std::endl;
    //timings
    std::chrono::steady_clock::time_point Tbegin, Tend;
    float f;
    float FPS[16];
    int i,Fcnt=0;

    cv::Mat image;
    lccv::PiCamera cam;
    cam.options->video_width=1024;
    cam.options->video_height=768;
    cam.options->framerate=3;
    //cam.options->verbose=true;
    //cv::namedWindow("Video",cv::WINDOW_NORMAL);
    cam.startVideo();
    int ch=0;
    while(ch!=27){
        Tbegin = std::chrono::steady_clock::now();
        if(!cam.getVideoFrame(image,1000)){
            std::cout<<"Timeout error"<<std::endl;
        }
        else{
            mb3.inference(image, objects);
            mb3.drawObjects(image, objects);

            Tend = std::chrono::steady_clock::now();
            f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
            if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
            for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
            putText(image, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

            cv::imshow("Video",image);

            ch=cv::waitKey(10);

        }
    }
    cam.stopVideo();
    cv::destroyWindow("Video");

    return 0;
}
