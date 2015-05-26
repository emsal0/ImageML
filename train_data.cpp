#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
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

int myrandom (int i) { return std::rand()%i;}

int main(int argc, char ** argv) {
    srand(time(0));
    std::vector<string> files;
    string dir("Eng-Characters/Img/GoodImg/Bmp/");
    getdir(dir,files);
    std::vector< std::pair<std::string,unsigned> > mlmap;

    for (unsigned i=0;i<files.size();i++) {
        string curr = files[i];
        int num;
        if (curr.size() > 3) {
            num = atoi(curr.substr(curr.size()-3).c_str()) - 1;
        }
        if (num < 10) {
            string path = dir + files[i];
            vector<string> inner_files;
            getdir(path,inner_files);
            for (unsigned j=0;j<inner_files.size();j++) {
                if (inner_files[j].size() > 2) {
                    string curr_inner = path + "/" + inner_files[j];
                    std::pair<std::string,unsigned> f;
                    f.first = curr_inner;
                    f.second = num;
                    mlmap.push_back(f);
                }
            }
        }
    }

    Net * my_net;
    if (!ifstream("test2.txt")) {
        vector<unsigned> topology;
        topology.push_back(16*16);
        topology.push_back(10);
        my_net = new Net(topology);
    } else {
        my_net = new Net("test2.txt");
    }
    std::random_shuffle(mlmap.begin(),mlmap.end(),myrandom);
    for (unsigned i=0; i<mlmap.size();i++) {
        cv::Mat image;
        vector<double> target_v(10,0);
        target_v[mlmap[i].second] = 1;
        image = cv::imread(mlmap[i].first,CV_LOAD_IMAGE_COLOR);
        if (image.rows < 16 || image.cols < 16) {
            cout << "too small!" << endl;
            continue;
        }
        vector<double> simplified = simplify(image);
        my_net->feed_forward(simplified);
        my_net->backprop(target_v);
    }
    my_net->save_to_file("test2.txt");

    delete my_net;

    return(0);
}
