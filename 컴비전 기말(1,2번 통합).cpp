#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 히스토그램 분석으로 적응적 임계값 결정 함수
int calculateAdaptiveThreshold(const Mat& image) {
    // 1. 히스토그램 계산
    vector<int> histogram(256, 0); // 256개의 빈도를 저장할 히스토그램 배열
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            histogram[image.at<uchar>(y, x)]++; // 각 픽셀값에 해당하는 빈도 증가
        }
    }

    // 2. Otsu's Method로 적응적 임계값 결정
    int totalPixels = image.rows * image.cols; // 이미지의 총 픽셀 수
    int sumB = 0, wB = 0, wF = 0, sum = 0;
    float maxVar = 0.0; // 클래스 간 최대 분산
    int threshold = 0;  // 최적의 임계값 저장 변수

    for (int t = 0; t < 256; t++) sum += t * histogram[t]; // 밝기 값의 총합 계산
    for (int t = 0; t < 256; t++) {
        wB += histogram[t]; // 백그라운드 누적 빈도
        if (wB == 0) continue; // 백그라운드가 없으면 스킵
        wF = totalPixels - wB; // 포그라운드 누적 빈도
        if (wF == 0) break; // 포그라운드가 없으면 종료

        sumB += t * histogram[t]; // 백그라운드의 밝기 값 합산
        float mB = static_cast<float>(sumB) / wB; // 백그라운드 평균 밝기
        float mF = static_cast<float>(sum - sumB) / wF; // 포그라운드 평균 밝기

        // 클래스 간 분산 계산
        float betweenVar = wB * wF * (mB - mF) * (mB - mF);
        if (betweenVar > maxVar) {
            maxVar = betweenVar; // 최대 분산 갱신
            threshold = t;       // 최적 임계값 갱신
        }
    }
    return threshold; // 최적 임계값 반환
}

// 에지 값을 양자화하여 원 컬러 영상에 반영
void applyEdgeValuesToColorImage(const Mat& colorImage, const Mat& edgeImage, Mat& resultImage) {
    // 에지 영상에서 최소값과 최대값 찾기
    double minVal, maxVal;
    minMaxLoc(edgeImage, &minVal, &maxVal); // 에지 영상의 최소/최대 밝기 값 계산

    // 결과 이미지 초기화
    resultImage = colorImage.clone(); // 원 컬러 영상 복사본 생성

    // 에지 값 양자화 및 컬러 영상 수정
    for (int y = 0; y < edgeImage.rows; y++) {
        for (int x = 0; x < edgeImage.cols; x++) {
            uchar edgeValue = edgeImage.at<uchar>(y, x); // 에지 영상의 픽셀값 가져오기

            if (edgeValue > 0) { // 에지 픽셀에만 처리
                // 0~99 범위로 양자화
                uchar quantizedValue = static_cast<uchar>(99 - ((edgeValue - minVal) / (maxVal - minVal)) * 99);

                // 컬러 영상의 각 채널에 양자화된 값 반영
                resultImage.at<Vec3b>(y, x) = Vec3b(quantizedValue, quantizedValue, quantizedValue);
            }
        }
    }
}

int main() {
    // (1) 컬러 영상 읽기
    Mat colorImage = imread("dog.bmp"); // 입력 이미지 로드
    if (colorImage.empty()) {
        cout << "Image not found!" << endl;
        return -1; // 이미지가 없으면 종료
    }

    // (2) 컬러 -> 그레이스케일 변환
    Mat grayImage;
    cvtColor(colorImage, grayImage, COLOR_BGR2GRAY); // 그레이스케일로 변환

    // (3) Sobel 에지 검출
    Mat gradX, gradY, edgeImage;
    Sobel(grayImage, gradX, CV_16S, 1, 0); // X 방향 에지 검출
    Sobel(grayImage, gradY, CV_16S, 0, 1); // Y 방향 에지 검출

    // 에지 값의 절대값을 계산하고 8비트 범위로 변환
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);

    // X 및 Y 방향 에지를 결합하여 최종 에지 이미지 생성
    addWeighted(gradX, 0.5, gradY, 0.5, 0, edgeImage);

    // (4) 히스토그램 분석으로 임계값 결정
    int adaptiveThreshold = calculateAdaptiveThreshold(edgeImage); // Otsu의 방법으로 임계값 계산
    cout << "Adaptive Threshold: " << adaptiveThreshold << endl;

    // (5) 임계화 수행
    Mat thresholdedImage;
    threshold(edgeImage, thresholdedImage, static_cast<double>(adaptiveThreshold), 255, THRESH_BINARY); // 임계화

    // (6) 에지 값을 원 컬러 영상에 반영
    Mat resultImage;
    applyEdgeValuesToColorImage(colorImage, edgeImage, resultImage); // 에지 값 반영

    // 결과 출력
    imshow("Original Image", colorImage);           // 원본 컬러 영상
    imshow("Gray Image", grayImage);               // 그레이스케일 영상
    imshow("Edge Image", edgeImage);               // 에지 영상
    imshow("Thresholded Edge Image", thresholdedImage); // 임계화된 에지 영상
    imshow("Modified Color Image", resultImage);   // 에지 값을 반영한 컬러 영상

    waitKey(0); // 키 입력 대기 후 종료
    return 0;
}
