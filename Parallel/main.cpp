#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <emmintrin.h>
using namespace std;

const int PIXELLIMIT = 255;

unsigned char* convertToThreeChannel(unsigned char* image, int width, int heigth) {
    unsigned char* newImage = new unsigned char[3 * width * heigth];
    int j = 0;
    for (int i = 0; i < width * heigth * 4; i += 4) {
        newImage[j] = image[i];
        newImage[j + 1] = image[i + 1];
        newImage[j + 2] = image[i + 2];
        j += 3;
    }
    return newImage;
}

unsigned char* negativeFilter(unsigned char* imageData, int width, int height) {
    unsigned char* newImage = new unsigned char[width * height * 3];
    int length = width * height * 3;

    for (int i = 0; i < length; i++) {
        newImage[i] = char(PIXELLIMIT - int(imageData[i]));
    }
    return newImage;
}

unsigned char* gaussFilter(unsigned char* image, int width, int height, int countChannel, double sigma, int kernelSize) {

    int halfOfKernelSize = kernelSize / 2;
    double* kernel = new double[kernelSize * kernelSize];
    double sum = 0;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {

            int x = i - halfOfKernelSize;
            int y = j - halfOfKernelSize;

            kernel[i * kernelSize + j] = exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
            sum += kernel[i * kernelSize + j];
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }

    unsigned char* newImage = new unsigned char[width * height * countChannel];

    for (int channel = 0; channel < countChannel; channel++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                double pixel = 0;

                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {

                        int pixelX = x + i - halfOfKernelSize;
                        int pixelY = y + j - halfOfKernelSize;

                        if (pixelX >= 0 and pixelX < width and pixelY >= 0 and pixelY < height) {
                            pixel += kernel[i * kernelSize + j] * image[(pixelY * width + pixelX) * countChannel + channel];
                        }
                    }
                }
                newImage[(y * width + x) * countChannel + channel] = pixel;
            }
        }
    }
    delete[] kernel;
    return newImage;
}

unsigned char* vectorNegativeFilter(unsigned char* imageData, int width, int height) {
    unsigned char* newImage = new unsigned char[width * height * 3];
    int length = width * height * 3;
    const __m128i pixelLimit = _mm_set1_epi8(PIXELLIMIT);

    for (int i = 0; i < length; i += 16) {
        __m128i pixel = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&imageData[i]));
        pixel = _mm_sub_epi8(pixelLimit, pixel);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&newImage[i]), pixel);
    }

    return newImage;
}


unsigned char* openMP_negativeFilter(unsigned char* imageData, int width, int height) {
    unsigned char* newImage = new unsigned char[width * height * 3];
    int length = width * height * 3;

    #pragma omp parallel for
    for (int i = 0; i < length; i++) {
        newImage[i] = char(PIXELLIMIT - int(imageData[i]));
    }
    return newImage;
}

unsigned char* openMP_gaussFilter(unsigned char* image, int width, int height, int countChannel, double sigma, int kernelSize) {

    int halfOfKernelSize = kernelSize / 2;
    double* kernel = new double[kernelSize * kernelSize];
    double sum = 0;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {

            int x = i - halfOfKernelSize;
            int y = j - halfOfKernelSize;

            kernel[i * kernelSize + j] = exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
            sum += kernel[i * kernelSize + j];
        }
    }
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }

    unsigned char* newImage = new unsigned char[width * height * countChannel];

    #pragma omp parallel for
    for (int channel = 0; channel < countChannel; channel++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                double pixel = 0;

                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {

                        int pixelX = x + i - halfOfKernelSize;
                        int pixelY = y + j - halfOfKernelSize;

                        if (pixelX >= 0 and pixelX < width and pixelY >= 0 and pixelY < height) {
                            pixel += kernel[i * kernelSize + j] * image[(pixelY * width + pixelX) * countChannel + channel];
                        }
                    }
                }
                newImage[(y * width + x) * countChannel + channel] = pixel;
            }
        }
    }
    delete[] kernel;
    return newImage;
}

int main() {

    const char* input = "";
    const char* negative = "negative.png";
    const char* gauss = "gauss.png";

    int width, height, channels;
    int number;

    cout << "Choose a picture: \n" << "1 - 300x300\n" << "2 - 400x400\n" << "3 - 500x500\n" << "4 - 600x600\n"
        << "5 - 950x950\n" << "6 - 2400x2400\n";

    cin >> number;
    switch (number) {
    case 1:
        input = "300x300.png";
        break;
    case 2:
        input = "400x400.png";
        break;
    case 3:
        input = "500x500.png";
        break;
    case 4:
        input = "600x600.png";
        break;
    case 5:
        input = "950x950.png";
        break;
    case 6:
        input = "2400x2400.png";
        break;
    default:
        cout << "Enter another name" << endl;
        string name;
        cin >> name;
        input = name.c_str();
        break;
    }

    unsigned char* imageData = stbi_load(input, &width, &height, &channels, 0);

    if (imageData == nullptr) {
        cout << "There is no such picture or you entered the wrong name. Try again." << endl;
        exit(EXIT_FAILURE);
    }

    if (channels > 3) {
        cout << "Wait. Photo editing in progress..." << endl;
        imageData = convertToThreeChannel(imageData, width, height);
        channels = 3;
    }

    double sum = 0;
    unsigned char* negativeImage;
    unsigned char* gaussImage;

    cout << "Choose a filter: \n" << "1 - Negative Filter\n" << "2 - Gaussian Blur\n" << "3 - OpenMP Negative Filter\n" << "4 - OpenMP Gaussian Blur\n"
        << "5 - Vector Negative Filter\n";
    int filter;
    cin >> filter;

    switch (filter) {
    case 1:
        sum = 0;

        for (int i = 0; i < 1000; i++) {

            auto begin = chrono::high_resolution_clock::now();

            negativeImage = negativeFilter(imageData, width, height);

            auto end = chrono::high_resolution_clock::now();
            auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);

            sum += elapsedMS.count() / 1000000.0;
        }

        stbi_write_png(negative, width, height, channels, negativeImage, 0);

        cout << "The middle time of Negative Filter: " << sum / 1000 << " s\n";
        break;

    case 2:
        sum = 0;

        for (int i = 0; i < 20; i++) {
            auto begin = chrono::steady_clock::now();

            gaussImage = gaussFilter(imageData, width, height, channels, 7.2, 22);

            auto end = chrono::steady_clock::now();
            auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);

            sum += elapsedMS.count() / 1000000.0;
        }

        stbi_write_png(gauss, width, height, channels, gaussImage, 0);
        cout << "The middle time of Gauss Filter: " << sum / 20 << " s\n";
        break;

    case 3:
        sum = 0;

        for (int i = 0; i < 1000; i++) {

            auto begin = chrono::high_resolution_clock::now();

            negativeImage = openMP_negativeFilter(imageData, width, height);

            auto end = chrono::high_resolution_clock::now();
            auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);
            /*cout << elapsedMS.count() / 1000000.0 << endl;*/
            sum += elapsedMS.count() / 1000000.0;
        }

        stbi_write_png(negative, width, height, channels, negativeImage, 0);

        cout << "The middle time of Negative Filter: " << sum / 1000 << " s\n";
        break;

    case 4:
        sum = 0;

        for (int i = 0; i < 100; i++) {
            auto begin = chrono::steady_clock::now();

            gaussImage = openMP_gaussFilter(imageData, width, height, channels, 7.2, 22);

            auto end = chrono::steady_clock::now();
            auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);

            sum += elapsedMS.count() / 1000000.0;
        }

        stbi_write_png(gauss, width, height, channels, gaussImage, 0);
        cout << "The middle time of Gauss Filter: " << sum / 100 << " s\n";
        break;

    case 5:
        sum = 0;

        for (int i = 0; i < 1000; i++) {

            auto begin = chrono::high_resolution_clock::now();

            negativeImage = vectorNegativeFilter(imageData, width, height);

            auto end = chrono::high_resolution_clock::now();
            auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);

            sum += elapsedMS.count() / 1000000.0;
        }

        stbi_write_png(negative, width, height, channels, negativeImage, 0);

        cout << "The middle time of Negative Filter: " << sum / 1000 << " s\n";
        break;

    default:
        cout << "Error" << endl;
        return 0;
    }

    cout << "Success" << endl;

    stbi_image_free(imageData);
    return 0;
}