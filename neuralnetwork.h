#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <assert.h>
#include <fstream>
using namespace std;

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
struct Connection {
    double weight;
    double delta_weight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
    public:
        Neuron(unsigned num_outputs,unsigned my_index);
        void set_output_value(double val);
        double get_output_value() const { return output_val; }
        void feed_forward(const Layer &prev_layer);
        void calc_output_gradients(double target_val);
        void calc_hidden_gradients(const Layer &next_layer);
        void update_input_weights(Layer &prev_layer);

    private:
        static double eta;
        static double alpha;
        static double transfer_function(double x);
        static double transfer_function_derivative(double x);
        static double random_weight() {return rand() / double (RAND_MAX); }
        double sum_DOW(const Layer &next_layer) const;
        double output_val;
        vector<Connection> output_weights;
        unsigned index;
        double gradient;
};


class Net {
    public:
        Net(const vector<unsigned> &topology);
        void feed_forward(const vector<double> &input_vals);
        void backprop(const vector<double> &target_vals);
        void get_results(vector<double> &result_vals) const;
        vector<unsigned> get_topology(); 
        void save_to_file(const char * filename);
    private:
        std::vector<Layer> layers; 
        double error;
        double recent_average_error;
        vector<unsigned> top;
};
#endif
