#include "process_image.h"

vector<unsigned> simplify(cv::Mat image) {
    vector<unsigned> ret;

    int height = image.rows;
    int width = image.cols;

    int small_height = height / 32;
    int small_width = width / 32;
   
    for (unsigned row=0; row<32; row++) {
        for (unsigned col=0;col<32;col++) {
            //cout << "row " << row << ", col " << col << endl;
            cv::Mat small_image = cv::Mat(image, cv::Rect(small_width*col,small_height*row,small_width,small_height));
            //vector<cv::Mat> channels;
            //cv::split(small_image,channels);

            cv::Scalar avg_color = cv::mean(small_image);
            double blue = avg_color[0];
            double green = avg_color[1];
            double red = avg_color[2];

            //int c = (255 - blue > blue) || (255 - green > green) || (255 - red > red);
            double h_avg = (blue + green + red) / 3;
            ret.push_back(255 - h_avg > h_avg);
        }
    }
    return ret;
} 
