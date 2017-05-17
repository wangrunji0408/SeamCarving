#include <iostream>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using std::cout;
using std::endl;
using std::cerr;

const int inf = 1 << 28;

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

std::vector<int> findMinColSeam (cv::Mat1b imat)
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
            ff += imat.at<uchar>(i, j);
        }
    std::vector<int> cols((unsigned)m);
    cols[m-1] = (int)(std::min_element(f[m], f[m] + n) - f[m]);
    for(int i=m-2; i>=0; --i)
        cols[i] = from[i][cols[i+1]];
    return cols;
}

cv::Mat1b makeIMat (cv::Mat const& image)
{
    cv::Mat i0, i1, i2, rst;
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

void printSeam (cv::Mat src, std::vector<int> const& cols)
{
    for(int i=0; i<cols.size(); ++i)
        src.at<cv::Vec3b>(i, cols[i]) = cv::Vec3b(0, 0, 255);
}

cv::Mat seamCarving (cv::Mat const& src, cv::MatSize size)
{
    auto image = src.clone();
    int colTimes = std::max(0, src.size[1] - size[1]);
    int rowTimes = std::max(0, src.size[0] - size[0]);
    for(int i=0; i<colTimes; ++i)
    {
        auto imat = makeIMat(image);
        auto cols = findMinColSeam(imat);
        image = cutCol(image, cols);
    }
    cv::Mat image2;
    cv::transpose(image, image2); image = image2;
    for(int i=0; i<rowTimes; ++i)
    {
        auto imat = makeIMat(image);
        auto cols = findMinColSeam(imat);
        image = cutCol(image, cols);
    }
    cv::transpose(image, image2); image = image2;
    return image;
}

int main (int argc, char** argv) {
    if(argc != 4)
    {
        cout << "usage: <filepath> <height> <width>" << endl;
        return 0;
    }
    char* file = argv[1];
    auto image = cv::imread(file);
    cerr << image.size[0] << 'x' << image.size[1] << endl;
    int size[] = {atoi(argv[2]), atoi(argv[3])};
    image = seamCarving(image, cv::MatSize(size));
    char filename[100];
    sprintf(filename, "./result/result_%d.jpg", (int)time(0));
    cv::imwrite(filename, image);
    cv::imshow("dst", image);
    cv::waitKey();
}