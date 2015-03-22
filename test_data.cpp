#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "neuralnetwork.h"
#include "process_image.h"

int main(int argc, char ** argv) {
    Net my_net(argv[2]);
    vector<unsigned> top = my_net.get_topology();
    for (unsigned i=0;i<top.size();i++) {
        cout << top[i] << endl;
    }
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);    
    vector <double> simplified = simplify(image); 
    unsigned r=0;
    for (vector<double>::iterator it = simplified.begin();it!=simplified.end();++it) {
        cout << (int) *it;
        r++;
        if (r==16) {
            cout << endl;
            r=0;
        }
    }
    cout << endl;
    //vector<double> simplified = simplify(image);
    my_net.feed_forward(simplified);
    vector<double> results;
    my_net.get_results(results);
    cout << "{ ";
    for (unsigned i=0;i<results.size();i++) {
        cout << results[i] << " ";
    }
    cout << " }" << endl;
    return(0);
}

