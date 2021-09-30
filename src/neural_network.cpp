// This code is a modification of the code from http://robotics.hobbizine.com/arduinoann.html


#include <arduino.h>
#include "neural_network.h"
#include <math.h>


NeuralNetwork::NeuralNetwork() {

}



void NeuralNetwork::initWeights() {
    
    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        for(int j = 0 ; j <= InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = 0.0 ;
            float Rando = float(random(100))/100;
            HiddenWeights[j*HiddenNodes + i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    for(int i = 0 ; i < OutputNodes ; i ++ ) {    
        for(int j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j*OutputNodes + i] = 0.0 ;  
            float Rando = float(random(100))/100;        
            OutputWeights[j*OutputNodes + i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
}




void NeuralNetwork::forward(const float Input[]){


/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = HiddenWeights[InputNodes*HiddenNodes + i];
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i];
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[HiddenNodes*OutputNodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        // OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        // Error += 0.5 * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
}




void NeuralNetwork::backward(const float Input[], const float Target[]){

    Error = 0;
// FORWARD

/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = HiddenWeights[InputNodes*HiddenNodes + i];
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i];
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[HiddenNodes*OutputNodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        Error += 0.33333 * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }


// END FORWARD




/******************************************************************
* Backpropagate errors to hidden layer
******************************************************************/

    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        float Accum = 0.0 ;
        for(int j = 0 ; j < OutputNodes ; j++ ) {
            Accum += OutputWeights[i*OutputNodes + j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
    }

/******************************************************************
* Update Inner-->Hidden Weights
******************************************************************/

    for(int i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes*HiddenNodes + i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        HiddenWeights[InputNodes*HiddenNodes + i] += ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HiddenNodes + i];
            HiddenWeights[j*HiddenNodes + i] += ChangeHiddenWeights[j*HiddenNodes + i] ;
        }
    }

/******************************************************************
* Update Hidden-->Output Weights
******************************************************************/

    for(int i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes*OutputNodes + i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes*OutputNodes + i] ;
        OutputWeights[HiddenNodes*OutputNodes + i] += ChangeOutputWeights[HiddenNodes*OutputNodes + i] ;
        for(int j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j*OutputNodes + i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j*OutputNodes + i] ;
            OutputWeights[j*OutputNodes + i] += ChangeOutputWeights[j*OutputNodes + i] ;
        }
    }
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