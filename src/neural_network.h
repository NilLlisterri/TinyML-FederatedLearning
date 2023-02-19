#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK


/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

static const int InputNodes = 650;
static const int HiddenNodes = 25;
static const int OutputNodes = 4;
static const float InitialWeightMax = 0.5;

typedef unsigned int uint;
static const uint hiddenWeightsAmt = (InputNodes + 1) * HiddenNodes;
static const uint outputWeightsAmt = (HiddenNodes + 1) * OutputNodes;

class NeuralNetwork {
    public:

        void initialize(float LearningRate, float Momentum);
        // ~NeuralNetwork();

        float forward(const float Input[], const float Target[]);
        float backward(const float Input[], const float Target[]);

        float* get_output();

        float* get_HiddenWeights();
        float* get_OutputWeights();

        float get_error();
        // float asd[500] = {};
        
    private:
        float Hidden[HiddenNodes] = {};
        float Output[OutputNodes] = {};
        float HiddenWeights[(InputNodes+1) * HiddenNodes] = {};
        float OutputWeights[(HiddenNodes+1) * OutputNodes] = {};
        float HiddenDelta[HiddenNodes] = {};
        float OutputDelta[OutputNodes] = {};
        float ChangeHiddenWeights[(InputNodes+1) * HiddenNodes] = {};
        float ChangeOutputWeights[(HiddenNodes+1) * OutputNodes] = {};

        float (*activation)(float);

        float Error;
        float LearningRate;
        float Momentum;
};


#endif
