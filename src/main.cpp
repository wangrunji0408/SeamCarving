#include <iostream>
#include <vector>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using std::cout;
using std::endl;
using std::cerr;
using std::string;

const int inf = 1 << 28;
const uchar KEEP = 1;
const uchar DELETE = 2;

template <class T>
inline bool updateMin (T& x, T const& a)
{
    if(a < x)
    {
        x = a;
        return true;
    }
    return false;
}

std::vector<int> findMinColSeam (cv::Mat1b imat, cv::Mat1b mask)
{
    const int MAXN = 4000;
    static int f[MAXN][MAXN];
    static int from[MAXN][MAXN];
    int m = imat.size[0], n = imat.size[1];
    assert(m+1 < MAXN && n+1 < MAXN);
    for(int j=0; j<n; ++j)
        f[0][j] = imat.at<uchar>(0, j);
    for(int i=1; i<m; ++i)
        for(int j=0; j<n; ++j)
        {
            int &ff = f[i][j];
            ff = inf;
            for(int k = j-1; k <= j+1; ++k)
                if(k >= 0 && k < n && updateMin(ff, f[i-1][k]))
                    from[i][j] = k;
            if(mask.at<uchar>(i, j) == KEEP)
                ff += 512;
            else if(mask.at<uchar>(i, j) == DELETE)
                ff += -512;
            else
                ff += imat.at<uchar>(i, j);
        }
    std::vector<int> cols((unsigned)m);
    cols[m-1] = (int)(std::min_element(f[m-1], f[m-1] + n) - f[m-1]);
    cerr << f[m-1][cols[m-1]] << endl;
    for(int i=m-2; i>=0; --i)
        cols[i] = from[i][cols[i+1]];
    return cols;
}

string kernelName;

cv::Mat1b makeIMat (cv::Mat const& image)
{
    cv::Mat rst;
    if(kernelName == "simple")
    {
        cv::Mat i0, i1, i2;
        cv::Mat k1 = cv::Mat::zeros(2, 2, CV_32F);
        k1.at<float>(0, 0) = 0;
        k1.at<float>(0, 1) = 0;
        k1.at<float>(1, 0) = -1;
        k1.at<float>(1, 1) = 1;

        cv::Mat k2 = cv::Mat::zeros(2, 2, CV_32F);
        k2.at<float>(0, 0) = 0;
        k2.at<float>(0, 1) = -1;
        k2.at<float>(1, 0) = 0;
        k2.at<float>(1, 1) = 1;

        cv::filter2D(image, i1, -1, k1);
        cv::filter2D(image, i2, -1, k2);
        i0 = abs(i1) + abs(i2);
        cv::cvtColor(i0, rst, CV_BGR2GRAY);
    }
    else if(kernelName == "sobel")
    {
        /// Generate grad_x and grad_y
        cv::Mat gray, grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        /// Gradient X
        cv::Scharr( image, grad_x, -1, 1, 0);
        cv::convertScaleAbs( grad_x, abs_grad_x );
        /// Gradient Y
        cv::Scharr( image, grad_y, -1, 0, 1);
        cv::convertScaleAbs( grad_y, abs_grad_y );
        /// Total Gradient (approximate)
        cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gray );
        cv::cvtColor( gray, rst, CV_RGB2GRAY);
    }
    else if(kernelName == "laplace")
    {
        cv::Mat temp;
        cv::Laplacian(image, temp, -1);
        cv::cvtColor(temp, rst, CV_BGR2GRAY);
    }
    else
        throw std::invalid_argument("No such kernel: " + kernelName);
    return rst;
}

cv::Mat cutCol (cv::Mat const& src, std::vector<int> const& cols)
{
    assert(cols.size() == src.size[0]);
    auto dst = cv::Mat(src.size[0], src.size[1] - 1, src.type());
    for(int i=0; i < src.size[0]; ++i)
    {
        int col = cols[i];
        memcpy(dst.data + dst.step[0] * i,
               src.data + src.step[0] * i,
               src.step[1] * col);
        memcpy(dst.data + dst.step[0] * i + dst.step[1] * col,
               src.data + src.step[0] * i + src.step[1] * (col + 1),
               src.step[1] * (src.size[1] - col - 1));
    }
    return dst;
}

cv::Mat addCol (cv::Mat const& src, std::vector<int> const& cols)
{
    assert(cols.size() == src.size[0]);
    auto dst = cv::Mat(src.size[0], src.size[1] + 1, src.type());
    for(int i=0; i < src.size[0]; ++i)
    {
        int col = cols[i];
        memcpy(dst.data + dst.step[0] * i,
               src.data + src.step[0] * i,
               src.step[1] * col);
        memcpy(dst.data + dst.step[0] * i + dst.step[1] * (col + 1),
               src.data + src.step[0] * i + src.step[1] * col,
               src.step[1] * (src.size[1] - col));
        memcpy(dst.data + dst.step[0] * i + dst.step[1] * col,
               src.data + src.step[0] * i + src.step[1] * col,
               src.step[1]);
    }
    return dst;
}

inline cv::Mat transpose (cv::Mat const& src)
{
    cv::Mat dst;
    cv::transpose(src, dst);
    return dst;
}

void printSeam (cv::Mat src, std::vector<int> const& cols, cv::Mat mask, bool trans = false)
{
    src = src.clone();
    for(int i=0; i<cols.size(); ++i)
        src.at<cv::Vec3b>(i, cols[i]) = cv::Vec3b(0, 0, 255);
    src.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int *pos){
        if(mask.at<uchar>(pos) == KEEP)
            pixel = pixel * 0.5 + cv::Vec3b(0, 255, 0) * 0.5;
        else if(mask.at<uchar>(pos) == DELETE)
            pixel = pixel * 0.5 + cv::Vec3b(0, 0, 255) * 0.5;
    });
    if(trans)
        src = transpose(src);
    cv::imshow("image", src);
    cv::waitKey(1);
}

bool show;

cv::Mat seamCarvingCol (cv::Mat const& image, cv::Mat& mask, bool trans = false)
{
    auto imat = makeIMat(image);
    auto cols = findMinColSeam(imat, mask);
    if(show)
        printSeam(image, cols, mask, trans);
    mask = cutCol(mask, cols);
    return cutCol(image, cols);
}

cv::Mat seamCarving (cv::Mat const& src, cv::MatSize size, cv::Mat const& mask0)
{
    auto image = src.clone();
    auto mask = mask0.clone();
    int colTimes = std::max(0, src.size[1] - size[1]);
    int rowTimes = std::max(0, src.size[0] - size[0]);
    for(; colTimes > 0 && rowTimes > 0; colTimes--, rowTimes--)
    {
        image = seamCarvingCol(image, mask, false);
        image = transpose(image);
        mask = transpose(mask);

        image = seamCarvingCol(image, mask, true);
        image = transpose(image);
        mask = transpose(mask);
    }
    if(colTimes > 0)
    {
        for(; colTimes > 0; colTimes--)
            image = seamCarvingCol(image, mask, false);
    }
    else
    {
        image = transpose(image);
        for(; rowTimes > 0; rowTimes--)
            image = seamCarvingCol(image, mask, true);
        image = transpose(image);
    }
    return image;
}

cv::Mat readMask (const char* path)
{
    auto mask = cv::imread(path);
    cv::Mat1b rst = cv::Mat::zeros(mask.size[0], mask.size[1], CV_8U);
    mask.forEach<cv::Vec3b>([&](cv::Vec3b const& pixel, const int *pos){
        const auto green = cv::Vec3b(0, 255, 0);
        const auto red = cv::Vec3b(0, 0, 255);
        if(pixel[1] > 128)
            rst.at<uchar>(pos) = KEEP;
        else if(pixel[2] > 128)
            rst.at<uchar>(pos) = DELETE;
    });
    return rst;
}

//DEFINE_string(filePath, true, "File path");
//DEFINE_uint32(height, true, "Image height");
//DEFINE_uint32(width, true, "Image width");
//DEFINE_bool(show, false, "If show procedure");
//DEFINE_string(maskPath, false, "Mask path");
//DEFINE_string(kernel, false, "Kernel name");


int main (int argc, char** argv) {

    if(argc < 4 || argc > 6)
    {
        cout << "usage: <file_path> <height> <width> [kernel] [show] [mask_path]" << endl;
        return 0;
    }

    char* filePath = argv[1];
    auto image = cv::imread(filePath);
    kernelName = argc >= 5? argv[4]: "simple";
    show = argc >= 6? argv[5][0] == '1': false;
    auto mask = argc >= 7? readMask(argv[6]): cv::Mat::zeros(image.size[0], image.size[1], CV_8U);
    assert(image.size == mask.size);

    char filename[100];
    int id = (int)time(0);
    sprintf(filename, "./result/result_%d_%s_energy.jpg", id, kernelName.c_str());
    cv::imwrite(filename, makeIMat(image));

    cerr << image.size[0] << 'x' << image.size[1] << endl;
    int size[] = {atoi(argv[2]), atoi(argv[3])};
    image = seamCarving(image, cv::MatSize(size), mask);

    sprintf(filename, "./result/result_%d_%s.jpg", id, kernelName.c_str());
    cv::imwrite(filename, image);
    cv::imshow("image", image);
    cv::waitKey();
}