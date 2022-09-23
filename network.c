#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "network.h"

// *********** ARRAY LIST CODE **********

array_list* init_array_list(void) {
    array_list* new_array_list = malloc(sizeof(array_list));
    if (new_array_list == NULL) {
        puts("Failed to create list.");
        return NULL;
    }
    new_array_list->start = malloc(sizeof(float));
    if (new_array_list->start == NULL) {
        puts("Failed to allocate spae for list items.");
        return NULL;
    }
    new_array_list->num_items = -1;
    return new_array_list;
}

void array_list_insert (array_list* a, float item) {
    if (a->num_items == -1) { // checks if array is initialized but no value has been set yet
        a->start[0] = item;
        a->num_items = 1;
        return;
    }
    a->num_items++;
    a->start = realloc( a->start, sizeof(float) * (size_t)(a->num_items) );
    if (a->start == NULL) {
        puts("Failed to allocate space for list items. List destroyed.");
        return;
    }
    a->start[a->num_items - 1] = item;
}

float array_list_get (array_list* a, int index) {
    return a->start[index];
}

void array_list_remove (array_list* a) {
    if (a->num_items == 0) {
        puts("Attempting to remove items from empty array. Process terminated.");
        return;
    }; 
    a->num_items--;
    a->start = realloc( a->start, sizeof(int) * (size_t)(a->num_items) );
    if (a->start == NULL) {
        puts("Failed to allocate space for list items. List destroyed.");
        return;
    }
}

void delete_array_list (array_list* a) {
    free(a->start);
    free(a);
}

// ******* END ARRAY LIST CODE *********

static float sigmoid (float x) {
    return pow(1 + exp(-x), -1);
}

static float feed_forward (neuron* n, float inputs[2]) {
    float w1 = n->weights[0];
    float w2 = n->weights[1];
    float x1 = inputs[0];
    float x2 = inputs[1];
    float b = n->bias;

    return sigmoid(w1*x1 + w2*x2 + b);
}

static neuron* init_neuron (float weights[2], float bias) {
    neuron* new_neuron = malloc(sizeof(neuron));
    new_neuron->weights[0] = weights[0];
    new_neuron->weights[1] = weights[1];
    new_neuron->bias = bias;
    return new_neuron;
}

network* init_network (void) {
    // create new neurons with random weights and biases for the new network
    neuron* new_neurons[3];
    for (int i = 0; i < sizeof(new_neurons)/sizeof(neuron*); i++) {
        float weights[] = { ( rand() % 10000 ) / 10000, ( rand() % 10000 ) / 10000 };
        float bias = ( rand() % 10000 ) / 10000;
        new_neurons[i] = init_neuron(weights, bias);
    }

    network* new_network = malloc(sizeof(network));
    new_network->h1 = new_neurons[0];
    new_network->h2 = new_neurons[1];
    new_network->o1 = new_neurons[2];  
    return new_network;
}

float network_feed_forward(network* n, float inputs[2]) {
    float out_h1 = feed_forward(n->h1, inputs);
    float out_h2 = feed_forward(n->h2, inputs);

    float o1_in[] = { out_h1, out_h2 };
    float out_o1 = feed_forward(n->o1, o1_in);
    return out_o1;
}

static float deriv_sigmoid (float x) {
    float fx = sigmoid(x);
    return fx * (1 - fx);
}

static float mse_loss (array_list* y_true, array_list* y_pred) {
    float total = 0;
    for (int i = 0; i < y_true->num_items; i ++) {
        total += pow( array_list_get(y_true, i) - array_list_get(y_pred, i), 2 );
    }
    return total / y_true->num_items;
}

void network_train (network* n, array_list* x_1, array_list* x_2, array_list* all_y_trues, int print) {

    float learn_rate = 0.1;
    int epochs = 1000;

    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < x_1->num_items; j++) {

            int x[] = { array_list_get(x_1, j), array_list_get(x_2, j) };

            float sum_h1 = n->h1->weights[0] * x[0] + n->h1->weights[1] * x[1] + n->h1->bias;
            float h1 = sigmoid(sum_h1);

            float sum_h2 = n->h2->weights[0] * x[0] + n->h2->weights[1] * x[1] + n->h2->bias;
            float h2 = sigmoid(sum_h2);

            float sum_o1 = n->o1->weights[0] * h1 + n->o1->weights[1] * h2 + n->o1->bias;
            float o1 = sigmoid(sum_o1);
            float y_pred = o1;

            // --- Calculate partial derivatives.
            // --- Naming: d_L_d_w1 represents "partial L / partial w1"
            int y_true = array_list_get(all_y_trues, j);
            float d_L_d_ypred = -2 * (y_true - y_pred);

            // Neuron o1
            float d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1);
            float d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1);
            float d_ypred_d_b3 = deriv_sigmoid(sum_o1);

            float d_ypred_d_h1 = n->o1->weights[0] * deriv_sigmoid(sum_o1);
            float d_ypred_d_h2 = n->o1->weights[1] * deriv_sigmoid(sum_o1);

            // Neuron h1
            float d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1);
            float d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1);
            float d_h1_d_b1 = deriv_sigmoid(sum_h1);

            // Neuron h2
            float d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2);
            float d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2);
            float d_h2_d_b2 = deriv_sigmoid(sum_h2);

            // --- Update weights and biases
            // Neuron h1
            n->h1->weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
            n->h1->weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
            n->h1->bias -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

            // Neuron h2
            n->h2->weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
            n->h2->weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
            n->h2->bias -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

            // Neuron o1
            n->o1->weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w5;
            n->o1->weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w6;
            n->o1->bias -= learn_rate * d_L_d_ypred * d_ypred_d_b3;

        }
        if (i % 10 == 0 && print == 1) {
            array_list* y_preds = init_array_list();
            for (int j = 0; j < x_1->num_items; j++) {
                float inputs[] = { (float)array_list_get(x_1, j), (float)array_list_get(x_2, j) };
                array_list_insert(y_preds, network_feed_forward(n, inputs));
            }
            float loss = mse_loss(all_y_trues, y_preds);
            printf("Epoch %i loss: %.3f\n", i, loss);
            array_list_remove(y_preds);
        }
    } 
}