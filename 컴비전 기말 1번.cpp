#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 히스토그램 분석으로 적응적 임계값 결정 함수
int calculateAdaptiveThreshold(const Mat& image) {
    // 히스토그램 계산
    vector<int> histogram(256, 0);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            histogram[image.at<uchar>(y, x)]++;
        }
    }

    // 임계값 결정: Otsu's Method 사용
    int totalPixels = image.rows * image.cols;
    int sumB = 0, wB = 0, wF = 0, sum = 0;
    float maxVar = 0.0;
    int threshold = 0;

    for (int t = 0; t < 256; t++) sum += t * histogram[t];
    for (int t = 0; t < 256; t++) {
        wB += histogram[t];  // 백그라운드 누적
        if (wB == 0) continue;
        wF = totalPixels - wB;  // 포그라운드 누적
        if (wF == 0) break;

        sumB += t * histogram[t];
        float mB = static_cast<float>(sumB) / wB;
        float mF = static_cast<float>(sum - sumB) / wF;

        float betweenVar = wB * wF * (mB - mF) * (mB - mF);
        if (betweenVar > maxVar) {
            maxVar = betweenVar;
            threshold = t;
        }
    }
    return threshold;
}

int main() {
    // (1) 컬러 영상 읽기
    Mat colorImage = imread("dog.bmp");
    if (colorImage.empty()) {
        cout << "Image not found!" << endl;
        return -1;
    }

    // (2) 컬러 -> 그레이스케일 변환
    Mat grayImage;
    cvtColor(colorImage, grayImage, COLOR_BGR2GRAY);

    // (3) Sobel 에지 검출
    Mat gradX, gradY, edgeImage;
    Sobel(grayImage, gradX, CV_16S, 1, 0);
    Sobel(grayImage, gradY, CV_16S, 0, 1);

    // Sobel 결과를 절대값으로 변환하여 8비트 범위로 매핑
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);

    // X 및 Y 방향 에지를 결합하여 최종 에지 이미지 생성
    addWeighted(gradX, 0.5, gradY, 0.5, 0, edgeImage);

    // (4) 히스토그램 분석으로 임계값 결정
    int adaptiveThreshold = calculateAdaptiveThreshold(edgeImage);
    cout << "Adaptive Threshold: " << adaptiveThreshold << endl;

    // (5) 임계화 수행
    Mat thresholdedImage;
    threshold(edgeImage, thresholdedImage, static_cast<double>(adaptiveThreshold), 255, THRESH_BINARY);

    // 결과 출력
    imshow("Original Image", colorImage);
    imshow("Gray Image", grayImage);
    imshow("Edge Image", edgeImage);
    imshow("Thresholded Edge Image", thresholdedImage);
    waitKey(0);

    return 0;
}
