#include "neuralnetwork.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron (unsigned num_outputs, unsigned my_index) {
    for (unsigned c=0;c<num_outputs;++c) {
        output_weights.push_back(Connection());
        output_weights.back().weight = random_weight();
        output_weights.back().delta_weight = 0.0;
    }
    index = my_index;
}

void Neuron::update_input_weights(Layer &prev_layer) {
    for (unsigned n=0;n<prev_layer.size();++n) {
        Neuron &neuron = prev_layer[n];
        double old_delta_weight = neuron.output_weights[index].delta_weight;

        double new_delta_weight = eta * neuron.get_output_value() * gradient + alpha * old_delta_weight;
        neuron.output_weights[index].delta_weight = new_delta_weight;
        neuron.output_weights[index].weight -= new_delta_weight;
    } 
}

double Neuron::transfer_function(double x) {
    return (1.0 / (1.0 + exp(-x)));
}

double Neuron::transfer_function_derivative(double x) {
    double z = Neuron::transfer_function(x);
    return z * (1-z);
}

void Neuron::feed_forward(const Layer &prev_layer) {
    double sum = 0.0;

    for (unsigned n=0; n<prev_layer.size(); ++n) {
        sum += prev_layer[n].get_output_value() * prev_layer[n].output_weights[index].weight;
    }
    output_val = Neuron::transfer_function(sum);
}

double Neuron::sum_DOW(const Layer &next_layer) const {
    double sum = 0.0;
    for (unsigned n=0; n<next_layer.size() - 1; ++n ) { 
        sum += output_weights[n].weight * next_layer[n].gradient;
    }
}

void Neuron::calc_hidden_gradients(const Layer &next_layer) {
    double dow = sum_DOW(next_layer);
    gradient = dow * Neuron::transfer_function_derivative(output_val);
}

void Neuron::calc_output_gradients(double target_val) {
    double delta = target_val - output_val;
    gradient = delta * Neuron::transfer_function_derivative(output_val);
}

void Neuron::set_output_value(double val) {
    output_val = val;
}


void Net::get_results(vector<double> &result_vals) const {
    result_vals.clear();
    for (unsigned n=0;n<layers.back().size() -1;++n) {
        result_vals.push_back(layers.back()[n].get_output_value());
    }
}

void Net::feed_forward(const vector<double> &input_vals) {
    assert(input_vals.size() == layers[0].size());
    for (unsigned i=0; i<input_vals.size();++i) {
        layers[0][i].set_output_value(input_vals[i]);
    }
    for (unsigned layer_num = 1; layer_num < layers.size(); ++layer_num) {
        Layer &prev_layer = layers[layer_num - 1];
        for (unsigned n = 0; n < layers[layer_num].size() - 1; ++n) {
            layers[layer_num][n].feed_forward(prev_layer);
        }
    }
}

void Net::backprop(const vector<double> &target_vals) {
    Layer &output_layer = layers.back(); 
    assert(target_vals.size() == output_layer.size());
    error = 0.0;
    for (unsigned n=0; n<output_layer.size() - 1;++n) {
        double delta = target_vals[n] - output_layer[n].get_output_value();
        error += delta*delta;
    }
    error /= output_layer.size() - 1;
    error = sqrt(error);

    for (unsigned n =0; n< output_layer.size() - 1; ++n) {
        output_layer[n].calc_output_gradients(target_vals[n]);
    }

    for (unsigned layer_num = layers.size() -2; layer_num > 0; --layer_num) {
        Layer &hidden_layer = layers[layer_num];
        Layer &next_layer = layers[layer_num+1];

        for (unsigned n=0;n<hidden_layer.size();++n) {
            hidden_layer[n].calc_hidden_gradients(next_layer);
        }
    }

    for (unsigned layer_num = layers.size() - 1; layer_num > 0; --layer_num) {
        Layer &layer = layers[layer_num];
        Layer &prev_layer = layers[layer_num -1];
        for (unsigned n=0;n<layer.size();++n) {
            layer[n].update_input_weights(prev_layer);
        }
    }
}

vector<unsigned> Net::get_topology() {
    return top;
}

void Net::save_to_file(const char * filename) {
     
    ofstream ofs(filename, std::ofstream::out);
    for (unsigned i=0;i<top.size();i++) {
        ofs << top[i] << " ";
    }
    ofs << endl;
    for (unsigned layer_num = 0; layer_num < layers.size(); layer_num++) {
        for (unsigned neuron_num = 0; neuron_num < layers[layer_num]; neuron_num++) {
            Neuron curr_neuron = layers[layer_num][neuron_num];
            for (unsigned weight_num = 0; weight_num < curr_neuron.output_weights.size()-1;weight_num++) {
                Connection curr_weight = curr_neuron.output_weights[weight_num];
                ofs << curr_weight.weight << " " << curr_weight.delta_weight << ", ";
            }

            Connection last_weight = curr_neuron.output_weights.back();
            ofs << last_weight.weight << " " << last_weight.delta_weight << endl;
        }          
    } 

    ofs.close();
}

Net::Net(const vector<unsigned> &topology) {
    vector<unsigned> x(topology);
    top = x;
    unsigned num_layers = topology.size();
    for (unsigned i = 0; i<num_layers; ++i) {
        layers.push_back(Layer());
        unsigned num_outputs = i == topology.size() - 1? 0:topology[i+1];

        for (unsigned neuron_num = 0; neuron_num <= topology[i]; ++neuron_num) {
            layers.back().push_back(Neuron(num_outputs,neuron_num));

        }
        layers.back().back().set_output_value(1.0);
    }
}

