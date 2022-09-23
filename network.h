#ifndef network_h
#define network_h

typedef struct {
    float weights[2];
    float bias;
} neuron;

typedef struct {
    neuron* h1;
    neuron* h2;
    neuron* o1;
} network;

typedef struct {
    float* start;
    int num_items;
} array_list;

array_list* init_array_list(void);
void array_list_insert (array_list* a, float item);
float array_list_get (array_list* a, int index);
void array_list_remove (array_list* a);
void delete_array_list (array_list* a);

network* init_network (void);
float network_feed_forward(network* n, float inputs[2]);
void network_train (network* n, array_list* x_1, array_list* x_2, array_list* all_y_trues, int print);

#endif