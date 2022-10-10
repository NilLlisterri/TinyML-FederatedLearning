// This code is a modification of the code from http://robotics.hobbizine.com/arduinoann.html


#include <arduino.h>
#include "neural_network.h"
#include <math.h>

void NeuralNetwork::initialize(float LearningRate, float Momentum) {
    this->LearningRate = LearningRate;
    this->Momentum = Momentum;
}

float NeuralNetwork::forward(const float Input[], const float Target[]){
    float error = 0;

    // Compute hidden layer activations
    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = HiddenWeights[InputNodes*HiddenNodes + i];
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i];
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

    // Compute output layer activations and calculate errors
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[HiddenNodes*OutputNodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        error += (1.0/OutputNodes) * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
    return error;
}

float NeuralNetwork::backward(const float Input[], const float Target[]){
    float error = 0;

    // Forward
    // Compute hidden layer activations
    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = HiddenWeights[InputNodes*HiddenNodes + i];
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i];
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

    // Compute output layer activations and calculate errors
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[HiddenNodes*OutputNodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        error += (1.0/OutputNodes) * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
    // End forward

    // Backward
    // Backpropagate errors to hidden layer
    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        float Accum = 0.0 ;
        for(int j = 0 ; j < OutputNodes ; j++ ) {
            Accum += OutputWeights[i*OutputNodes + j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
    }

    // Update Inner-->Hidden Weights
    for(int i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes*HiddenNodes + i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        HiddenWeights[InputNodes*HiddenNodes + i] += ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HiddenNodes + i];
            HiddenWeights[j*HiddenNodes + i] += ChangeHiddenWeights[j*HiddenNodes + i] ;
        }
    }

    // Update Hidden-->Output Weights
    for(int i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes*OutputNodes + i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes*OutputNodes + i] ;
        OutputWeights[HiddenNodes*OutputNodes + i] += ChangeOutputWeights[HiddenNodes*OutputNodes + i] ;
        for(int j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j*OutputNodes + i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j*OutputNodes + i] ;
            OutputWeights[j*OutputNodes + i] += ChangeOutputWeights[j*OutputNodes + i] ;
        }
    }

    return error;
}


float* NeuralNetwork::get_output(){
    return Output;
}

float* NeuralNetwork::get_HiddenWeights(){
    return HiddenWeights;
}

float* NeuralNetwork::get_OutputWeights(){
    return OutputWeights;
}

float NeuralNetwork::get_error(){
    return Error;
}