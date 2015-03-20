#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "neuralnetwork.h"
#include "process_image.h"

using namespace std;

int main(int argc, char ** argv) {
    srand(time(0));

    cv::Mat image;
    image = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
    
    vector <unsigned> simplified = simplify(image); 
    unsigned r=0;
    for (vector<unsigned>::iterator it = simplified.begin();it!=simplified.end();++it) {
        cout << *it;
        r++;
        if (r==32) {
            cout << endl;
            r=0;
        }
    }
    cout << endl;
    return(0);
}