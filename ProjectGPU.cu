#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #include <cuda_runtime.h>

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef int BOOL;
typedef int INT;
typedef double REAL;

#define FALSE 0
#define TRUE 1
#define NOT !
#define AND &&
#define OR ||

#define MIN_REAL -HUGE_VAL
#define MAX_REAL +HUGE_VAL

#define sqr(x) ((x) * (x))

#define LO 0.1
#define HI 0.9
#define BIAS 1

// Structure for a single layer in the neural network
typedef struct {
    int units;          // Number of units in the layer
    REAL *output;       // Output of each unit in the layer
    REAL *error;        // Error of each unit in the layer
    REAL *weights;      // Weight matrix (including bias weights)
    REAL *dWeights;     // Weight changes for momentum
} LAYER;

// Structure for the neural network
typedef struct {
    int numLayers;      // Number of layers in the network
    LAYER *layers;      // Array of layers
    REAL *inputs;       // Input values
    REAL *desiredOutputs;   // Desired output values
    REAL *outputs;      // Output values
} NET;

#define MAX_LAYERS 5

#define sqr(x) ((x) * (x))

// Functions for initializing, propagating, backpropagating, and adjusting weights for a layer
void PropagateLayerCUDA(NET *net, LAYER *lower, LAYER *upper);
void BackpropagateLayerCUDA(NET *net, LAYER *upper, LAYER *lower);
void AdjustWeightsCUDA(NET *net, LAYER *upper, LAYER *lower);

// Function to create a neural network
NET *CreateNet(int numLayers, int layerSizes[]);

// Function to free the memory used by a neural network
void DestroyNet(NET *net);

// Function to set the inputs of the neural network
void SetInputs(NET *net, REAL *inputs);

// Function to set the desired outputs of the neural network
void SetDesiredOutputs(NET *net, REAL *outputs);

// Function to get the outputs of the neural network
REAL *GetOutputs(NET *net);

// Function to train the neural network using backpropagation
void TrainNet(NET *net, REAL *inputs, REAL *outputs, REAL eta, REAL alpha);

// Function to compute the error of the neural network
REAL GetError(NET *net);

// Function to print the weights of the neural network
void PrintWeights(NET *net);

int main() {
    // Initialize the random number generator
    srand(time(NULL));

    // Define the structure of the neural network
    int layerSizes[MAX_LAYERS] = {2, 2, 1};
    NET *net = CreateNet(3, layerSizes);

    // Define the training data
    REAL inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    REAL outputs[4] = {0, 1, 1, 0};

    // Train the neural network
    for (int epoch = 0; epoch < 10000; epoch++) {
        for (int i = 0; i < 4; i++) {
            SetInputs(net, inputs[i]);
            SetDesiredOutputs(net, &outputs[i]);
            TrainNet(net, inputs[i], &outputs[i], 0.5, 0.1);
        }
        if (epoch % 1000 == 0) {
            REAL error = GetError(net);
            printf("Epoch %d, Error = %f\n", epoch, error);
        }
    }

    // Print the weights of the neural network
    PrintWeights(net);

    // Clean up and free memory
    DestroyNet(net);

    return 0;
}

__global__ void PropagateLayerKernel(REAL *lowerOutput, REAL *upperWeights, REAL *upperOutput, int lowerUnits, int upperUnits,
                                     REAL gain) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < upperUnits) {
        REAL sum = 0;
        for (int j = 0; j <= lowerUnits; j++) {
            sum += upperWeights[idx * (lowerUnits + 1) + j] * lowerOutput[j];
        }
        upperOutput[idx] = 1 / (1 + exp(-gain * sum));
    }
}

__global__ void BackpropagateLayerKernel(REAL *lowerOutput, REAL *upperWeights, REAL *upperError, REAL *lowerError,
                                        int lowerUnits, int upperUnits, REAL gain) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < lowerUnits) {
        REAL out = lowerOutput[idx];
        REAL err = 0;
        for (int j = 1; j <= upperUnits; j++) {
            err += upperWeights[j * (lowerUnits + 1) + idx] * upperError[j];
        }
        lowerError[idx] = gain * out * (1 - out) * err;
    }
}

__global__ void AdjustWeightsKernel(REAL *lowerOutput, REAL *upperError, REAL *weights, REAL *dWeights, int lowerUnits,
                                    int upperUnits, REAL eta, REAL alpha) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i <= upperUnits && j <= lowerUnits) {
        REAL Out = lowerOutput[j];
        REAL Err = upperError[i];
        REAL dWeight = dWeights[i * (lowerUnits + 1) + j];
        weights[i * (lowerUnits + 1) + j] += eta * Err * Out + alpha * dWeight;
        dWeights[i * (lowerUnits + 1) + j] = eta * Err * Out;
    }
}

void PropagateLayerCUDA(NET *net, LAYER *lower, LAYER *upper) {
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (upper->units + threadsPerBlock - 1) / threadsPerBlock;

    // Launch PropagateLayerKernel on GPU
    PropagateLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(lower->output, upper->weights, upper->output, lower->units,
                                                             upper->units, 1.0);
    cudaDeviceSynchronize();
}

void BackpropagateLayerCUDA(NET *net, LAYER *upper, LAYER *lower) {
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (lower->units + threadsPerBlock - 1) / threadsPerBlock;

    // Launch BackpropagateLayerKernel on GPU
    BackpropagateLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(lower->output, upper->weights, upper->error,
                                                                 lower->error, lower->units, upper->units, 1.0);
    cudaDeviceSynchronize();
}

void AdjustWeightsCUDA(NET *net, LAYER *upper, LAYER *lower) {
    // Kernel configuration
    int threadsPerBlockX = 16;
    int threadsPerBlockY = 16;
    int blocksPerGridX = (upper->units + threadsPerBlockX - 1) / threadsPerBlockX;
    int blocksPerGridY = (lower->units + threadsPerBlockY - 1) / threadsPerBlockY;

    dim3 blocksPerGrid(blocksPerGridX, blocksPerGridY);

    // Launch AdjustWeightsKernel on GPU
    AdjustWeightsKernel<<<blocksPerGrid, dim3(threadsPerBlockX, threadsPerBlockY)>>>(lower->output, upper->error,
                                                                                     upper->weights, upper->dWeights,
                                                                                     lower->units, upper->units, 0.5,
                                                                                     0.1);
    cudaDeviceSynchronize();
}

NET *CreateNet(int numLayers, int layerSizes[]) {
    int i, j;
    NET *net = (NET *)malloc(sizeof(NET));

    net->numLayers = numLayers;
    net->layers = (LAYER *)malloc(numLayers * sizeof(LAYER));
    net->inputs = (REAL *)malloc(layerSizes[0] * sizeof(REAL));
    net->desiredOutputs = (REAL *)malloc(layerSizes[numLayers - 1] * sizeof(REAL));
    net->outputs = (REAL *)malloc(layerSizes[numLayers - 1] * sizeof(REAL));

    for (i = 0; i < numLayers; i++) {
        net->layers[i].units = layerSizes[i];
        net->layers[i].output = (REAL *)malloc(layerSizes[i] * sizeof(REAL));
        net->layers[i].error = (REAL *)malloc(layerSizes[i] * sizeof(REAL));
        if (i > 0) {
            net->layers[i].weights = (REAL *)malloc((layerSizes[i - 1] + 1) * layerSizes[i] * sizeof(REAL));
            net->layers[i].dWeights = (REAL *)malloc((layerSizes[i - 1] + 1) * layerSizes[i] * sizeof(REAL));
            for (j = 0; j < layerSizes[i - 1] + 1; j++) {
                for (int k = 0; k < layerSizes[i]; k++) {
                    net->layers[i].weights[j * layerSizes[i] + k] = ((REAL)rand() / RAND_MAX - 0.5) * 2.0 * 0.5;
                    net->layers[i].dWeights[j * layerSizes[i] + k] = 0.0;
                }
            }
        }
    }
    return net;
}

void DestroyNet(NET *net) {
    int i;
    for (i = 0; i < net->numLayers; i++) {
        free(net->layers[i].output);
        free(net->layers[i].error);
        if (i > 0) {
            free(net->layers[i].weights);
            free(net->layers[i].dWeights);
        }
    }
    free(net->layers);
    free(net->inputs);
    free(net->desiredOutputs);
    free(net->outputs);
    free(net);
}

void SetInputs(NET *net, REAL *inputs) {
    for (int i = 0; i < net->layers[0].units; i++) {
        net->inputs[i] = inputs[i];
    }
}

void SetDesiredOutputs(NET *net, REAL *outputs) {
    for (int i = 0; i < net->layers[net->numLayers - 1].units; i++) {
        net->desiredOutputs[i] = outputs[i];
    }
}

REAL *GetOutputs(NET *net) {
    return net->outputs;
}

void TrainNet(NET *net, REAL *inputs, REAL *outputs, REAL eta, REAL alpha) {
    // Set inputs and desired outputs
    SetInputs(net, inputs);
    SetDesiredOutputs(net, outputs);

    // Forward propagation
    for (int i = 1; i < net->numLayers; i++) {
        PropagateLayerCUDA(net, &net->layers[i - 1], &net->layers[i]);
    }

    // Backpropagation
    for (int i = net->numLayers - 1; i >= 1; i--) {
        BackpropagateLayerCUDA(net, &net->layers[i], &net->layers[i - 1]);
        AdjustWeightsCUDA(net, &net->layers[i], &net->layers[i - 1]);
    }
}

REAL GetError(NET *net) {
    REAL error = 0.0;
    for (int i = 0; i < net->layers[net->numLayers - 1].units; i++) {
        REAL delta = net->desiredOutputs[i] - net->outputs[i];
        error += 0.5 * delta * delta;
    }
    return error;
}

void PrintWeights(NET *net) {
    for (int i = 1; i < net->numLayers; i++) {
        int upperUnits = net->layers[i].units;
        int lowerUnits = net->layers[i - 1].units;
        printf("Layer %d Weights:\n", i);
        for (int j = 0; j < upperUnits; j++) {
            for (int k = 0; k <= lowerUnits; k++) {
                printf("%f ", net->layers[i].weights[j * (lowerUnits + 1) + k]);
            }
            printf("\n");
        }
    }
}
