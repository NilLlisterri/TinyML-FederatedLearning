#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK





/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

static const int PatternCount = 3;
static const int InputNodes = 650;
static const int HiddenNodes = 20;
static const int OutputNodes = 3;
static const float InitialWeightMax = 0.5;


class NeuralNetwork {
    public:

        NeuralNetwork(float LearningRate, float Momentum);
        // ~NeuralNetwork();

        void initWeights();
        void forward(const float Input[]);
        void backward(const float Input[], const float Target[]);

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

       
        float Error;
        float LearningRate;
        float Momentum;
};


#endif
