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

using namespace std;
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

int main(int argc, char ** argv) {
    srand(time(0));
    vector<string> files;
    //string dir(argv[1]); 
//    getdir(dir,files);
    int target = atoi(argv[2]);
    vector<double> target_v(10,0);
    target_v[target] = 1;
 
    for (unsigned i=0;i<target_v.size();i++) {
        cout << target_v[i] << " ";
    }
    cout << endl;

    Net * my_net;
    if (!ifstream("test.txt")) {
        vector<unsigned> topology;
        topology.push_back(16*16);
        topology.push_back(16*16);
        topology.push_back(16*16);
        topology.push_back(10); 
        my_net = new Net(topology);
    } else {
        my_net = new Net("test.txt");
    }

  /*  for (unsigned i=0;i<files.size();i++) {
        string cf = dir+files[i];
        cv::Mat image;
        cout << cf << endl;
        image = cv::imread(cf, CV_LOAD_IMAGE_COLOR);    
        cout << image.rows << " " << image.cols << endl;
        if (image.rows < 16 || image.cols < 16) {
            cout << "too small!" << endl;
            continue;
        }
        vector<double> simplified = simplify(image);
        my_net->feed_forward(simplified);
        my_net->backprop(target_v);
    }*/

    cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
    if (image.rows < 16 || image.cols < 16) {
        cout << "IMAGE TOO SMALL" << endl;
        return -1;
    }
    vector<double> simplified = simplify(image);
    my_net->feed_forward(simplified);
    cout << "fed forward" << endl;
    my_net->backprop(target_v);
    cout << "backpropagated" << endl;
    my_net->save_to_file("test.txt");
    delete my_net;

    return(0);
}
