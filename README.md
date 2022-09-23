# simple-ml-model
Simple machine learning model built from a [tutorial by Victor Zhou](https://victorzhou.com/blog/intro-to-neural-networks/) written in Python, I implemented it in C.

Implements the following functions:

Array List Functions:

```
array_list* init_array_list(void);
  // allocate space for array list - does not yet store anything
void array_list_insert (array_list* a, float item);
  // add values to array list (must be done one-by-one)
float array_list_get (array_list* a, int index); 
  // retrieve the item at index 'index' of array list pointed to by 'a'. Equivalent to a->start[index]
void array_list_remove (array_list* a); 
  // deletes last item in array list pointed to by 'a' and frees memory associated with it
void delete_array_list (array_list* a); 
  // deletes entire list and frees all memory of list and member items
```

Neural Network Functions:
```
network* init_network (void);
  // initializes network
float network_feed_forward(network* n, float inputs[2]);
  // returns output from network 'n' resulting from inputs, used after network is trained 
void network_train (network* n, array_list* x_1, array_list* x_2, array_list* all_y_trues, int print);
  // network_train - trains neural network on data with two inputs (list x_1 and list x_2) with true outputs all_y_trues. Set 'print' argument to 1 to
                     print loss to stdout line after every ten training cycles.
```

Array List Functions used to create a dynamically-sized array for storing inputs to and outputs from the network. Can only store floats.
Array list must be initialized prior to insertion.

Neural Network takes two inputs and produces one output. There are two input neurons, a hidden layer of two neurons, and one output neuron.
Input neurons take data from lists x_1 and x_2, respectively.

Program was tested on sample height (x_1) and weight (x_2) data and used to predict gender (output).

Sample implementation with data provided in tutorial:

```
// main.c

#include <stdio.h>
#include "network.h"

int main (void) {

    int x_1_vals[] = { -2, 25, 17, -15 };
    // 135 subtracted from weight values
    int x_2_vals[] = { -1, 6, 4, -6 };
    // 66 subtracted from height values
    int y_true_vals[] = { 1, 0, 0, 1 };

    array_list* x_1 = init_array_list();
    array_list* x_2 = init_array_list();
    array_list* y_trues = init_array_list();

    for (int i = 0; i < sizeof(x_1_vals)/sizeof(int); i++) {
        array_list_insert(x_1, x_1_vals[i]);
        array_list_insert(x_2, x_2_vals[i]);
        array_list_insert(y_trues, y_true_vals[i]);
    }

    network* network = init_network();
    network_train(network, x_1, x_2, y_trues, 0);
    float in[] = { 20, 2 };
    float pred = network_feed_forward(network, in);
    printf("%.3f", pred);
}
```
