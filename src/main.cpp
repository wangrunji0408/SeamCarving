#include <iostream>
#include <vector>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

const int inf = 1 << 20;
const uchar KEEP = 1;
const uchar DELETE = 2;

typedef vector<vector<int>> Seams;

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

template <class T>
inline T mean (T const& a, T const& b)
{
    return a + (b - a) / 2;
}

const int MAXN = 4000;
int f[MAXN][MAXN];

void dp(cv::Mat1i const& imat) {
    int m = imat.size[0];
    int n = imat.size[1];
    assert(m + 1 < MAXN && n + 1 < MAXN);
    for(int j=0; j<n; ++j)
        f[0][j] = imat.at<int>(0, j);
    for(int i=1; i<m; ++i)
        for(int j=0; j<n; ++j)
        {
            int &ff = f[i][j];
            const int* begin = &f[i-1][std::max(0, j - 1)];
            const int* end   = &f[i-1][std::min(n, j + 2)];
            ff = *std::min_element(begin, end);
            ff += imat.at<int>(i, j);
        }
}

Seams findMinColSeams (cv::Mat1i const& imat0, size_t k)
{
    cv::Mat1i imat = imat0.clone();
    int m = imat.size[0];
    int n = imat.size[1];

    Seams seams;
    seams.reserve(k);
    while(k--)
    {
        dp(imat);
        vector<int> cols((size_t)m);
        cols[m-1] = (int)(std::min_element(f[m-1], f[m-1] + n) - f[m-1]);
        auto fmin = f[m-1][cols[m-1]];
        if(fmin >= inf)
            break;
        cerr << fmin << endl;
        for(int i=m-2; i>=0; --i) {
            int j = cols[i + 1];
            const int* begin = &f[i][std::max(0, j-1)];
            const int* end   = &f[i][std::min(n, j+2)];
            cols[i] = int(std::min_element(begin, end) - f[i]);
        }
        for(int i=0; i<m; ++i) {
            int const& col = cols[i];
            imat.at<int>(i, col) = inf;
            imat.at<int>(i, std::max(0, col - 1)) = inf;
            imat.at<int>(i, std::min(n-1, col + 1)) = inf;
        }
        seams.push_back(cols);
    }
    return seams;
}

vector<int> findMinColSeam (cv::Mat1i const& imat)
{
    return findMinColSeams(imat, 1)[0];
}

cv::Mat1i applyMask (cv::Mat1b const& energy, cv::Mat1b const& mask)
{
    assert(energy.size == mask.size);
    cv::Mat1i rst = cv::Mat::zeros(mask.size[0], mask.size[1], CV_32S);
    rst.forEach([&](int& pixel, const int *pos){
        if(mask.at<uchar>(pos) == KEEP)
            pixel = 512;
        else if(mask.at<uchar>(pos) == DELETE)
            pixel = -512;
        else
            pixel = energy.at<uchar>(pos);
    });
    return rst;
}

string kernelName;

cv::Mat1i makeIMat (cv::Mat const& image)
{
    cv::Mat1b gray;
    cv::Mat1i rst;
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    if(kernelName == "simple")
    {
        cv::Mat i0, i1, i2;
        cv::Mat k1 = cv::Mat::zeros(3, 3, CV_32F);
        k1.at<float>(1, 0) = -1;
        k1.at<float>(1, 2) = 1;

        cv::Mat k2 = cv::Mat::zeros(3, 3, CV_32F);
        k2.at<float>(0, 1) = -1;
        k2.at<float>(2, 1) = 1;

        cv::filter2D(gray, i1, CV_16S, k1);
        cv::filter2D(gray, i2, CV_16S, k2);
        rst = abs(i1) + abs(i2);
    }
    else if(kernelName == "sobel")
    {
        /// Generate grad_x and grad_y
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        /// Gradient X
        cv::Scharr( gray, grad_x, CV_16S, 1, 0);
        cv::convertScaleAbs( grad_x, abs_grad_x );
        /// Gradient Y
        cv::Scharr( gray, grad_y, CV_16S, 0, 1);
        cv::convertScaleAbs( grad_y, abs_grad_y );
        /// Total Gradient (approximate)
        cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, rst );
    }
    else if(kernelName == "laplace")
    {
        cv::Mat temp;
        cv::Laplacian(gray, temp, CV_16S, 3);
        rst = abs(temp);
    }
    else
        throw std::invalid_argument("No such kernel: " + kernelName);
    return rst;
}

inline void matcpy (cv::Mat const& src, cv::Mat const& dst, int i, int j1, int j2, int n)
{
    memcpy(dst.data + dst.step[0] * i + dst.step[1] * j2,
           src.data + src.step[0] * i + src.step[1] * j1,
           src.step[1] * n);
}

cv::Mat handleSeams (cv::Mat const& src, Seams const& seams, bool add = false)
{
    auto m = src.size[0];
    auto n = src.size[1];
    auto ns = seams.size();
    auto n1 = add? n + ns: n - ns;

    for(auto const& seam: seams)
        assert(seam.size() == m);

    auto dst = cv::Mat(m, (int) n1, src.type());

    vector<int> ids(ns); // 按seam从右到左排序
    for(int i=0; i<ns; ++i)
        ids[i] = i;
    std::sort(ids.begin(), ids.end(),
                [&](int const& i, int const& j) -> bool
                { return seams[i][0] < seams[j][0]; });

    // 按seam从左到右将src拷贝到dst
    auto zeros = vector<int>(m, 0);
    auto vecns = vector<int>(m, n);
    vector<int> const* lastSeam = &zeros;
    for(int cnt=0; cnt<=ns; ++cnt)
    {
        auto nowSeam = cnt == ns? &vecns: &seams[ids[cnt]];
        for(int i=0; i < m; ++i)
        {
            int col0 = (*lastSeam)[i];
            int col1 = (*nowSeam)[i];
            assert(col0 < col1 || (col0 == 0 && col1 == 0));
            typedef cv::Vec3b T;
            if(add) {
                matcpy(src, dst, i, col0, col0 + cnt, col1 - col0);
                if(col0 != 0)
                    dst.at<T>(i, col0 + cnt-1) = mean(src.at<T>(i, col0), src.at<T>(i, col0 - 1));
            }
            else {
                if(cnt == 0)
                    matcpy(src, dst, i, 0, 0, col1);
                else
                    matcpy(src, dst, i, col0 + 1, col0 - cnt+1, col1 - col0 - 1);
            }
        }
        lastSeam = nowSeam;
    }
    return dst;
}

inline cv::Mat cutSeams (cv::Mat const& src, Seams const& seams)
{
    return handleSeams(src, seams, false);
}

inline cv::Mat addSeams (cv::Mat const& src, Seams const& seams)
{
    return handleSeams(src, seams, true);
}

inline cv::Mat transpose (cv::Mat const& src)
{
    cv::Mat dst;
    cv::transpose(src, dst);
    return dst;
}

void printSeams (cv::Mat src, Seams const& seams, cv::Mat const& mask, bool trans, const char* name)
{
    src = src.clone();
    auto dcolor = 200 / seams.size();
    for(int k=0; k<seams.size(); ++k) {
        auto const& cols = seams[k];
        for (int i = 0; i < cols.size(); ++i)
            src.at<cv::Vec3b>(i, cols[i]) = cv::Vec3b(0, 0, (uchar) (255 - k * dcolor));
    }
    src.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int *pos){
        if(mask.at<uchar>(pos) == KEEP)
            pixel = pixel * 0.5 + cv::Vec3b(0, 255, 0) * 0.5;
        else if(mask.at<uchar>(pos) == DELETE)
            pixel = pixel * 0.5 + cv::Vec3b(0, 0, 255) * 0.5;
    });
    if(trans)
        src = transpose(src);
    cv::imshow(name, src);
    cv::waitKey(1);
}

void printSeam (cv::Mat src, vector<int> const& cols, cv::Mat const& mask, bool trans)
{
    printSeams(src, Seams(1, cols), mask, trans, "image");
}

bool show;

cv::Mat seamCarvingCol (cv::Mat const& image, cv::Mat& mask, bool trans = false)
{
    auto imat0 = makeIMat(image);
    auto imat = applyMask(imat0, mask);
    auto seams = findMinColSeams(imat, 1);
    if(show) {
        printSeams(image, seams, mask, trans, "image");
//        cv::Mat mat0, mat1;
//        imat.convertTo(mat0, CV_8U);
//        cv::cvtColor(mat0, mat1, CV_GRAY2BGR);
//        printSeams(mat1, seams, mask, trans, "energy");
//        cv::Mat1i mat = imat.clone();
//        mat.forEach([&](int& pixel, const int* pos){
//            pixel = f[pos[0]][pos[1]];
//        });
//        mat.convertTo(mat0, CV_8U);
//        cv::cvtColor(mat0, mat1, CV_GRAY2BGR);
//        printSeams(mat1, seams, mask, trans, "f");
    }
    mask = cutSeams(mask, seams);
    return cutSeams(image, seams);
}

cv::Mat seamCarvingEnlargeCol (cv::Mat const& image, size_t seamSize, bool trans = false)
{
    if(seamSize == 0)
        return image;
    auto imat = makeIMat(image);
    auto seams = findMinColSeams(imat, seamSize);
    auto mask = cv::Mat::zeros(image.size[0], image.size[1], CV_8U);
    if(show)
        printSeams(image, seams, mask, trans, "image");
    return addSeams(image, seams);
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
        mask = transpose(mask);
        for(; rowTimes > 0; rowTimes--)
            image = seamCarvingCol(image, mask, true);
        image = transpose(image);
    }
    return image;
}

cv::Mat seamCarvingEnlarge (cv::Mat const& src, cv::MatSize size)
{
    auto image = src.clone();
    int colTimes = std::max(0, size[1] - src.size[1]);
    int rowTimes = std::max(0, size[0] - src.size[0]);
    int seamSize = colTimes / 2;
    while(size[1] - image.size[1] > 0)
        image = seamCarvingEnlargeCol(image, (size_t) std::min(seamSize, colTimes), false);
    seamSize = rowTimes / 2;
    image = transpose(image);
    while(size[0] - image.size[1] > 0)
        image = seamCarvingEnlargeCol(image, (size_t) std::min(seamSize, rowTimes), true);
    image = transpose(image);
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

    if(argc < 4 || argc > 7)
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
    sprintf(filename, "../result/result_%d_%s_energy.jpg", id, kernelName.c_str());
    cv::imshow("image", image);
    cv::waitKey();

    cerr << image.size[0] << 'x' << image.size[1] << endl;
    int size[] = {atoi(argv[2]), atoi(argv[3])};
    if(size[0] >= image.size[0] && size[1] >= image.size[1])
        image = seamCarvingEnlarge(image, cv::MatSize(size));
    else
        image = seamCarving(image, cv::MatSize(size), mask);

    sprintf(filename, "../result/result_%d_%s.jpg", id, kernelName.c_str());
    cv::imwrite(filename, image);
    cv::imshow("image", image);
    cv::waitKey();
}