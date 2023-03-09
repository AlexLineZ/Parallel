#pragma GCC optimize("03")
#pragma GCC target("avx2")
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
using namespace std;

const int PIXELLIMIT = 255;

unsigned char* negativeFilter(unsigned char* imageData, int width, int height) {
    unsigned char* newImage = new unsigned char[width * height * 3];
    int length = width * height * 3;
    for (int i = 0; i < length; i++) {
        newImage[i] = char(PIXELLIMIT - int(imageData[i]));
    }
    return newImage;
}

unsigned char* convertToThreeChannel(unsigned char* image, int width, int heigth) {
    unsigned char* newImage = new unsigned char[3 * width * heigth];
    int j = 0;
    for (int i = 0; i < width * heigth * 4; i+=4) {
        newImage[j] = image[i];
        newImage[j + 1] = image[i + 1];
        newImage[j + 2] = image[i + 2];
        j += 3;
    }
    return newImage;
}

unsigned char* gaussFilter(unsigned char* image, int width, int height, int countChannel, float sigma) {

    int kernelSize = ceil(sigma * 3) * 2 + 1;
    int halfOfKernelSize = kernelSize / 2;
    float* kernel = new float[kernelSize * kernelSize];

    float sum = 0;
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

    //#pragma omp parallel for
    for (int channel = 0; channel < countChannel; channel++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                float pixel = 0;

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


    auto begin = chrono::steady_clock::now();

    unsigned char* gaussImage = gaussFilter(imageData, width, height, channels, 7);
    stbi_write_png(gauss, width, height, channels, gaussImage, 0);

    auto end = chrono::steady_clock::now();

    auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);
    cout << "The time of Gauss Filter: " << elapsedMS.count() / 1000000.0 << " s\n";


    begin = chrono::steady_clock::now();

    unsigned char* negativeImage = negativeFilter(imageData, width, height);
	stbi_write_png(negative, width, height, channels, negativeImage, 0);

    end = chrono::steady_clock::now();
    elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);
    cout << "The time of Negative Filter: " << elapsedMS.count() / 1000000.0 << " s\n";

	cout << "Success" << endl;
	stbi_image_free(imageData);

}