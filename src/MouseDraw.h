//
// Created by 王润基 on 2017/6/13.
//

#ifndef INC_3SEAMCARVING_MOUSEDRAW_H
#define INC_3SEAMCARVING_MOUSEDRAW_H

#include <opencv2/highgui.hpp>

struct MouseArgs {
    cv::Mat3b const& image;
    cv::Mat3b showImage;
    cv::Mat1b mask;
    bool drawing, erasing;

    MouseArgs(const cv::Mat3b &image) : image(image)
    {
        drawing = false;
        erasing = false;
        showImage = image.clone();
        mask = cv::Mat::zeros(image.size[0], image.size[1], CV_8U);
    }
};

void mouseDrawCallback(int event, int x, int y, int flags, void *param)
{
    std::swap(x, y);
    auto mouseArgs = (MouseArgs*)param;
    int r = 10;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:  mouseArgs->drawing = true;  break;
        case CV_EVENT_LBUTTONUP:  mouseArgs->drawing = false;   break;
        case CV_EVENT_RBUTTONDOWN:  mouseArgs->erasing = true;  break;
        case CV_EVENT_RBUTTONUP:  mouseArgs->erasing = false;   break;
        default: break;
    }
    if(event == CV_EVENT_MOUSEMOVE && mouseArgs->drawing) // draw
    {
        for(int i = -r; i <= r; ++i)
            for(int j = -r; j <= r; ++j)
                if(i * i + j * j <= r * r)
                {
                    int x0 = x + i, y0 = y + j;
                    if(!(x0 >= 0 && x0 < mouseArgs->image.size[0] && y0 >= 0 && y0 < mouseArgs->image.size[1]))
                        continue;
                    mouseArgs->mask.at<u8>(x0, y0) = 255;
                    mouseArgs->showImage.at<cv::Vec3b>(x0, y0) = cv::Vec3b(0, 128, 0);
                }
        cv::imshow("Draw ROI", mouseArgs->showImage);
    }
    else if(event == CV_EVENT_MOUSEMOVE && mouseArgs->erasing) // erase
    {
        for(int i = -r; i <= r; ++i)
            for(int j = -r; j <= r; ++j)
                if(i * i + j * j <= r * r)
                {
                    int x0 = x + i, y0 = y + j;
                    if(!(x0 >= 0 && x0 < mouseArgs->image.size[0] && y0 >= 0 && y0 < mouseArgs->image.size[1]))
                        continue;
                    mouseArgs->mask.at<u8>(x0, y0) = 0;
                    mouseArgs->showImage.at<cv::Vec3b>(x0, y0) = mouseArgs->image.at<cv::Vec3b>(x0, y0);
                }
        cv::imshow("Draw ROI", mouseArgs->showImage);
    }
}

cv::Mat getMaskShowWindow (cv::Mat const& image)
{
    MouseArgs mouseArgs(image);
    cv::namedWindow("Draw ROI",CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback("Draw ROI", mouseDrawCallback, (void*)&mouseArgs);
    cv::imshow("Draw ROI", mouseArgs.showImage);
    cv::waitKey();
    while(cv::waitKey(100) != 27) // esc
    {
//        if(mouseArgs.drawing || mouseArgs.erasing)
//            cv::imshow("Draw ROI", mouseArgs.showImage);
    }
    cv::destroyWindow("Draw ROI");
    return mouseArgs.mask;
}

#endif //INC_3SEAMCARVING_MOUSEDRAW_H
