#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// ������׷� �м����� ������ �Ӱ谪 ���� �Լ�
int calculateAdaptiveThreshold(const Mat& image) {
    // ������׷� ���
    vector<int> histogram(256, 0);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            histogram[image.at<uchar>(y, x)]++;
        }
    }

    // �Ӱ谪 ����: Otsu's Method ���
    int totalPixels = image.rows * image.cols;
    int sumB = 0, wB = 0, wF = 0, sum = 0;
    float maxVar = 0.0;
    int threshold = 0;

    for (int t = 0; t < 256; t++) sum += t * histogram[t];
    for (int t = 0; t < 256; t++) {
        wB += histogram[t];  // ��׶��� ����
        if (wB == 0) continue;
        wF = totalPixels - wB;  // ���׶��� ����
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
    // (1) �÷� ���� �б�
    Mat colorImage = imread("dog.bmp");
    if (colorImage.empty()) {
        cout << "Image not found!" << endl;
        return -1;
    }

    // (2) �÷� -> �׷��̽����� ��ȯ
    Mat grayImage;
    cvtColor(colorImage, grayImage, COLOR_BGR2GRAY);

    // (3) Sobel ���� ����
    Mat gradX, gradY, edgeImage;
    Sobel(grayImage, gradX, CV_16S, 1, 0);
    Sobel(grayImage, gradY, CV_16S, 0, 1);

    // Sobel ����� ���밪���� ��ȯ�Ͽ� 8��Ʈ ������ ����
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);

    // X �� Y ���� ������ �����Ͽ� ���� ���� �̹��� ����
    addWeighted(gradX, 0.5, gradY, 0.5, 0, edgeImage);

    // (4) ������׷� �м����� �Ӱ谪 ����
    int adaptiveThreshold = calculateAdaptiveThreshold(edgeImage);
    cout << "Adaptive Threshold: " << adaptiveThreshold << endl;

    // (5) �Ӱ�ȭ ����
    Mat thresholdedImage;
    threshold(edgeImage, thresholdedImage, static_cast<double>(adaptiveThreshold), 255, THRESH_BINARY);

    // ��� ���
    imshow("Original Image", colorImage);
    imshow("Gray Image", grayImage);
    imshow("Edge Image", edgeImage);
    imshow("Thresholded Edge Image", thresholdedImage);
    waitKey(0);

    return 0;
}
