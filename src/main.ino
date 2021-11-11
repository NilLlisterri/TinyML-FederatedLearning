// This program uses several functionalities and modifications 
// from the EdgeImpulse inferencing library.

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0


/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <training_kws_inference.h>
#include "neural_network.h"

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal


const uint8_t button_1 = 2;
const uint8_t button_2 = 3;
const uint8_t button_3 = 4;
const uint8_t button_4 = 5;
//uint8_t num_button = 0; // 0 represents none
bool button_pressed = false;

// Defaults: 0.3, 0.9
static NeuralNetwork myNetwork;
const float threshold = 0.6;

uint16_t num_epochs = 0;


/**
 * @brief      Arduino setup function
 */
void setup() {
    Serial.begin(9600);

    // Start button_configuration
    pinMode(button_1, INPUT);
    pinMode(button_2, INPUT);
    pinMode(button_3, INPUT);
    pinMode(button_4, INPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

    digitalWrite(LED_BUILTIN, HIGH);

    if (microphone_setup(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }

    init_network_model();
    digitalWrite(LED_BUILTIN, LOW);

    num_epochs = 0;
}


void init_network_model() {
    char startChar;
    do {
        startChar = Serial.read();
        Serial.println("Waiting for new model...");
    } while(startChar != 's'); // s -> START

    Serial.println("start");
    float learningRate = readFloat();
    float momentum = readFloat();

    myNetwork.initialize(learningRate, momentum);

    char* myHiddenWeights = (char*) myNetwork.get_HiddenWeights();
    for (uint16_t i = 0; i < (InputNodes+1) * HiddenNodes; ++i) {
        Serial.write('n');
        while(Serial.available() < 4) {}
        for (int n = 0; n < 4; n++) {
            myHiddenWeights[i*4] = Serial.read();
        }
    }

    char* myOutputWeights = (char*) myNetwork.get_OutputWeights();
    for (uint16_t i = 0; i < (HiddenNodes+1) * OutputNodes; ++i) {
        Serial.write('n');
        while(Serial.available() < 4) {}
        for (int n = 0; n < 4; n++) {
            myOutputWeights[i*4+n] = Serial.read();
        }
    }

    Serial.println("Received new model.");
}

float readFloat() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    return *(float *)&res;
}

void train(int nb, bool only_forward) {
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei::matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

    // for(int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
    //     Serial.print(inference.buffer[i]);
    //     Serial.print(",");
    // }
    // Serial.println();

    EI_IMPULSE_ERROR r = get_one_second_features(&signal, &features_matrix, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to get features (%d)\n", r);
        return;
    }


    float myTarget[3] = {0};
    myTarget[nb-1] = 1.f; // button 1 -> {1,0,0};  button 2 -> {0,1,0};  button 3 -> {0,0,1}

    float backward_error = 0;
    if (!only_forward) {
        // BACKWARD
        backward_error = myNetwork.backward(features_matrix.buffer, myTarget);
        ++num_epochs;
    }

    // FORWARD
    float forward_error = myNetwork.forward(features_matrix.buffer, myTarget);

    float error = forward_error;
    if (!only_forward) {
        error = backward_error;
    }

    float* myOutput = myNetwork.get_output();

    //uint8_t num_button_output = 0;
    //float max_output = 0.f;
    // Serial.print("Inference result: ");


    // Info to plot & graph!
    Serial.println("graph");

    // Print outputs
    for (size_t i = 0; i < 3; i++) {
        ei_printf_float(myOutput[i]);
        Serial.print(" ");
    //    if (myOutput[i] > max_output && myOutput[i] > threshold) {
    //        num_button_output = i + 1;
    //    }
    }
    Serial.print("\n");

    // Print error
    ei_printf_float(error);
    Serial.print("\n");

    Serial.println(num_epochs, DEC);

    char* myError = (char*) &error;
    Serial.write(myError, sizeof(float));
    
    Serial.println(nb, DEC);
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    digitalWrite(LEDR, HIGH);           // OFF
    digitalWrite(LEDG, HIGH);           // OFF
    digitalWrite(LEDB, HIGH);           // OFF
    digitalWrite(LED_BUILTIN, HIGH);    // ON

    uint8_t num_button = 0;

    bool only_forward = false;

    if (digitalRead(button_1) == HIGH && (button_pressed == false || num_button == 1)) {
        digitalWrite(LEDR, LOW);  //  ON
        num_button = 1;
        button_pressed = true;
        
        while(digitalRead(button_1) == HIGH) {}
        delay(200);
    }
    else if (digitalRead(button_2) == HIGH && (button_pressed == false || num_button == 2)) {
        digitalWrite(LEDG, LOW);  //  ON
        num_button = 2;
        button_pressed = true;

        while(digitalRead(button_2) == HIGH) {}
        delay(200);
    }
    else if (digitalRead(button_3) == HIGH && (button_pressed == false || num_button == 3)) {
        digitalWrite(LEDB, LOW);  //  ON
        num_button = 3;
        button_pressed = true;

        while(digitalRead(button_3) == HIGH) {}
        delay(200);
    }
    else if (digitalRead(button_4) == HIGH && (button_pressed == false || num_button == 4)) {
        digitalWrite(LEDR, LOW);  //  ON
        digitalWrite(LEDG, LOW);  //  ON
        digitalWrite(LEDB, LOW);  //  ON

        Serial.println("start_fl");
        // only_forward = true;

        // Debounce
        while(digitalRead(button_4) == HIGH) {}
        delay(200);
    } 
    
    if (button_pressed == true) {
        Serial.println("Recording...");
        bool m = microphone_inference_record();
        if (!m) {
            Serial.println("ERR: Failed to record audio...");
            return;
        }
        Serial.println("Recording done");

        train(num_button, only_forward);

        button_pressed = false;
    } else {
        int read = Serial.read();
        if (read == '>') { // s -> FEDERATED LEARNING
            /***********************
             * Federate Learning
             ***********************/
            Serial.write('<');
            digitalWrite(LED_BUILTIN, HIGH);    // ON
            delay(1000);
            if (Serial.read() == 's') {
                Serial.println("start");
                Serial.println(num_epochs);
                num_epochs = 0;

                /*******
                 * Sending model
                 *******/

                // Sending hidden layer
                char* myHiddenWeights = (char*) myNetwork.get_HiddenWeights();
                for (uint16_t i = 0; i < (InputNodes+1) * HiddenNodes; ++i) {
                    Serial.write(myHiddenWeights+i*sizeof(float), sizeof(float));
                }

                // Sending output layer
                char* myOutputWeights = (char*) myNetwork.get_OutputWeights();
                for (uint16_t i = 0; i < (HiddenNodes+1) * OutputNodes; ++i) {
                    Serial.write(myOutputWeights+i*sizeof(float), sizeof(float));
                }

                /*****
                 * Receiving model
                 *****/
                // Receiving hidden layer
                for (uint16_t i = 0; i < (InputNodes+1) * HiddenNodes; ++i) {
                    Serial.write('n');
                    while(Serial.available() < 4) {}
                    for (int n = 0; n < 4; n++) {
                        myHiddenWeights[i*4+n] = Serial.read();
                    }
                }

                // Receiving output layer
                for (uint16_t i = 0; i < (HiddenNodes+1) * OutputNodes; ++i) {
                    Serial.write('n');
                    while(Serial.available() < 4) {}
                    for (int n = 0; n < 4; n++) {
                        myOutputWeights[i*4+n] = Serial.read();
                    }
                }
            }

            digitalWrite(LED_BUILTIN, LOW);    // OFF
        } else if (read == 't') { // Train with a sample
            Serial.println("ok");

            while(Serial.available() < 1) {}
            uint8_t num_button = Serial.read();
            Serial.print("Button "); Serial.println(num_button);

            while(Serial.available() < 1) {}
            bool only_forward = Serial.read() == 1;
            Serial.print("Only forward "); Serial.println(only_forward);
            
            byte ref[2];
            for(int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
                while(Serial.available() < 2) {}
                Serial.readBytes(ref, 2);
                inference.buffer[i] = 0;
                inference.buffer[i] = (ref[1] << 8) | ref[0];
            }
            Serial.print("Sample received for button ");
            Serial.println(num_button);
            train(num_button, only_forward);
        }
    }
}


void ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.write(print_buf);
    }
}


static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead>>1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}


static bool microphone_setup(uint32_t n_samples) {
    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    // optionally set the gain, defaults to 20
    PDM.setGain(80);
    PDM.setBufferSize(4096);

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        PDM.end();
        return false;
    }
    return true;
}


static bool microphone_inference_record(void) {
    inference.buf_ready = 0;
    inference.buf_count = 0;
    while(inference.buf_ready == 0) {
        delay(10);
    }
    return true;
}


static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}
