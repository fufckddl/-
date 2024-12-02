#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// ������׷� �м����� ������ �Ӱ谪 ���� �Լ�
int calculateAdaptiveThreshold(const Mat& image) {
    // 1. ������׷� ���
    vector<int> histogram(256, 0); // 256���� �󵵸� ������ ������׷� �迭
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            histogram[image.at<uchar>(y, x)]++; // �� �ȼ����� �ش��ϴ� �� ����
        }
    }

    // 2. Otsu's Method�� ������ �Ӱ谪 ����
    int totalPixels = image.rows * image.cols; // �̹����� �� �ȼ� ��
    int sumB = 0, wB = 0, wF = 0, sum = 0;
    float maxVar = 0.0; // Ŭ���� �� �ִ� �л�
    int threshold = 0;  // ������ �Ӱ谪 ���� ����

    for (int t = 0; t < 256; t++) sum += t * histogram[t]; // ��� ���� ���� ���
    for (int t = 0; t < 256; t++) {
        wB += histogram[t]; // ��׶��� ���� ��
        if (wB == 0) continue; // ��׶��尡 ������ ��ŵ
        wF = totalPixels - wB; // ���׶��� ���� ��
        if (wF == 0) break; // ���׶��尡 ������ ����

        sumB += t * histogram[t]; // ��׶����� ��� �� �ջ�
        float mB = static_cast<float>(sumB) / wB; // ��׶��� ��� ���
        float mF = static_cast<float>(sum - sumB) / wF; // ���׶��� ��� ���

        // Ŭ���� �� �л� ���
        float betweenVar = wB * wF * (mB - mF) * (mB - mF);
        if (betweenVar > maxVar) {
            maxVar = betweenVar; // �ִ� �л� ����
            threshold = t;       // ���� �Ӱ谪 ����
        }
    }
    return threshold; // ���� �Ӱ谪 ��ȯ
}

// ���� ���� ����ȭ�Ͽ� �� �÷� ���� �ݿ�
void applyEdgeValuesToColorImage(const Mat& colorImage, const Mat& edgeImage, Mat& resultImage) {
    // ���� ���󿡼� �ּҰ��� �ִ밪 ã��
    double minVal, maxVal;
    minMaxLoc(edgeImage, &minVal, &maxVal); // ���� ������ �ּ�/�ִ� ��� �� ���

    // ��� �̹��� �ʱ�ȭ
    resultImage = colorImage.clone(); // �� �÷� ���� ���纻 ����

    // ���� �� ����ȭ �� �÷� ���� ����
    for (int y = 0; y < edgeImage.rows; y++) {
        for (int x = 0; x < edgeImage.cols; x++) {
            uchar edgeValue = edgeImage.at<uchar>(y, x); // ���� ������ �ȼ��� ��������

            if (edgeValue > 0) { // ���� �ȼ����� ó��
                // 0~99 ������ ����ȭ
                uchar quantizedValue = static_cast<uchar>(99 - ((edgeValue - minVal) / (maxVal - minVal)) * 99);

                // �÷� ������ �� ä�ο� ����ȭ�� �� �ݿ�
                resultImage.at<Vec3b>(y, x) = Vec3b(quantizedValue, quantizedValue, quantizedValue);
            }
        }
    }
}

int main() {
    // (1) �÷� ���� �б�
    Mat colorImage = imread("dog.bmp"); // �Է� �̹��� �ε�
    if (colorImage.empty()) {
        cout << "Image not found!" << endl;
        return -1; // �̹����� ������ ����
    }

    // (2) �÷� -> �׷��̽����� ��ȯ
    Mat grayImage;
    cvtColor(colorImage, grayImage, COLOR_BGR2GRAY); // �׷��̽����Ϸ� ��ȯ

    // (3) Sobel ���� ����
    Mat gradX, gradY, edgeImage;
    Sobel(grayImage, gradX, CV_16S, 1, 0); // X ���� ���� ����
    Sobel(grayImage, gradY, CV_16S, 0, 1); // Y ���� ���� ����

    // ���� ���� ���밪�� ����ϰ� 8��Ʈ ������ ��ȯ
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);

    // X �� Y ���� ������ �����Ͽ� ���� ���� �̹��� ����
    addWeighted(gradX, 0.5, gradY, 0.5, 0, edgeImage);

    // (4) ������׷� �м����� �Ӱ谪 ����
    int adaptiveThreshold = calculateAdaptiveThreshold(edgeImage); // Otsu�� ������� �Ӱ谪 ���
    cout << "Adaptive Threshold: " << adaptiveThreshold << endl;

    // (5) �Ӱ�ȭ ����
    Mat thresholdedImage;
    threshold(edgeImage, thresholdedImage, static_cast<double>(adaptiveThreshold), 255, THRESH_BINARY); // �Ӱ�ȭ

    // (6) ���� ���� �� �÷� ���� �ݿ�
    Mat resultImage;
    applyEdgeValuesToColorImage(colorImage, edgeImage, resultImage); // ���� �� �ݿ�

    // ��� ���
    imshow("Original Image", colorImage);           // ���� �÷� ����
    imshow("Gray Image", grayImage);               // �׷��̽����� ����
    imshow("Edge Image", edgeImage);               // ���� ����
    imshow("Thresholded Edge Image", thresholdedImage); // �Ӱ�ȭ�� ���� ����
    imshow("Modified Color Image", resultImage);   // ���� ���� �ݿ��� �÷� ����

    waitKey(0); // Ű �Է� ��� �� ����
    return 0;
}
