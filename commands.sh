#!/bin/sh
g++ train_data.cpp neuralnetwork2.cpp process_image.cpp -ggdb `pkg-config --cflags opencv` -o train_data `pkg-config --libs opencv`
g++ neuralnetwork2.cpp process_image.cpp test_data.cpp -ggdb `pkg-config --cflags opencv` -o test_data `pkg-config --libs opencv`


