import serial
from serial.tools.list_ports import comports

import struct
import time
import numpy as np
import matplotlib.pyplot as plt


def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword:
            break
        else:
            print(f'({arduino.port}):',msg, end='')



def init_network(hidden_layer, output_layer, arduino):
    arduino.reset_input_buffer()
    arduino.write(b's')
    print_until_keyword('start', arduino)
    for i in range(len(hidden_layer)):
        arduino.read() # wait until confirmation of float received
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        arduino.write(data)
    
    for i in range(len(output_layer)):
        arduino.read() # wait until confirmation of float received
        float_num = output_layer[i]
        data = struct.pack('f', float_num)
        arduino.write(data)

    


def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except:
            print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            #port = "COM3";
            return serial.Serial(port, 9600)
        except:
            print(f"ERROR: Wrong port connection ({port})")


def plot_graph(graph_data):
    devices =  [x[2] for x in graph_data]
    for device_index in devices:
        epoch = [x[0] for x in graph_data if x[2] == device_index]
        error = [x[1] for x in graph_data if x[2] == device_index]
    
        plt.plot(epoch, error, label=device_index)
    plt.xlim(xmin=0.0) 
    plt.ylim(ymin=0.0)
    plt.ylabel('Loss') # or Error
    plt.xlabel('Epoch')
   
    plt.show()
    plt.autoscale()
    plt.pause(0.1)




# def main():
plt.ion()
plt.show()
num_devices = read_number("Number of devices: ")

available_ports = comports()
print("Available ports:")
for available_port in available_ports:
    print(available_port)


devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]

size_hidden_layer = (650+1)*16
size_output_layer = (16+1)*3

np.random.seed(12345)
hidden_layer = np.random.uniform(-0.5,0.5, (650+1)*16).astype('float32')
output_layer = np.random.uniform(-0.5, 0.5, (16+1)*3).astype('float32')

# To load a Pre-trained model
# hidden_layer = np.load("./hidden_montserrat.npy")
# output_layer = np.load("./output_montserrat.npy")

for d in devices:
    init_network(hidden_layer, output_layer, d)

devices_connected = devices

graph = [] # Modified to graph
################
# Infinite loop
################
while True:

    federated_learning_interval = 10
    federated_learning_interval = 100000000 # Modified to graph

    for d in devices:
        d.timeout = 0
        # d.timeout = None # Modified to graph

    countdown_print = [20,15,10, 5, 4, 3, 2, 1] 
    ini_time = time.time()

    while True:
        for d_index, d in enumerate(devices):
            msg = d.readline().decode()
            if (len(msg) > 0):
                print(f'({d.port}):', msg, end='')
                # Modified to graph
                if msg[:-2] == 'graph':
                    ne = d.readline()[:-2];
                    print("ne: ", ne);
                    n_epooch = int(ne)

                    #n_error = d.read(4)
                    #[n_error] = struct.unpack('f', n_error)
                    err = d.readline()[:-2]
                    print("ERr: ", err)
                    n_error = float(err)

                    nb = d.readline()[:-2]
                    #print("Nb: ", nb)
                    #n_button = int(nb)
                    graph.append([n_epooch, n_error, d_index])

                    plot_graph(graph)
                    print(graph)

        countdown = federated_learning_interval - int(time.time() - ini_time)
        if countdown <= 0:
            break
        elif countdown in countdown_print:
            print(f'Starting FL in {countdown} seconds')
            countdown_print.remove(countdown)


    print('Starting Federated Learning')
    old_devices_connected = devices_connected
    devices_connected = []
    devices_hidden_layer = np.empty((0,size_hidden_layer), dtype='float32')
    devices_output_layer = np.empty((0,size_output_layer), dtype='float32')
    devices_num_epochs = []

    ##################
    # Receiving models
    ##################
    for d in devices:
        d.reset_input_buffer()
        d.reset_output_buffer()
        d.timeout = 5

        print(f'Starting connection to {d.port} ...') # Hanshake
        d.write(b'>') # Python --> SYN --> Arduino
        if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
            d.write(b's') # Python --> ACK --> Arduino
            
            print('Connection accepted.')
            devices_connected.append(d)
            devices_hidden_layer = np.vstack((devices_hidden_layer, np.empty(size_hidden_layer)))
            devices_output_layer = np.vstack((devices_output_layer, np.empty(size_output_layer)))
            d.timeout = None

            print_until_keyword('start', d)
            devices_num_epochs.append(int(d.readline()[:-2]))

            print(f'Receiving model from {d.port} ...')
            ini_time = time.time()

            for i in range((650+1)*16): # hidden layer
                data = d.read(4)
                [float_num] = struct.unpack('f', data)
                devices_hidden_layer[-1][i] = float_num

            for i in range((16+1)*3): # output layer
                data = d.read(4)
                [float_num] = struct.unpack('f', data)
                devices_output_layer[-1][i] = float_num

            print(f'Model received from {d.port} ({time.time()-ini_time} seconds)')

            # if it was not connected before, we dont use the devices' model
            if not d in old_devices_connected:
                devices_num_epochs[-1] = 0
                print(f'Model not used. The device {d.port} has an outdated model')



        else:
            print(f'Connection timed out. Skipping {d.port}.')

    
    ####################
    # Processing models
    ####################

    # if sum == 0, any device made any epoch
    if sum(devices_num_epochs) > 0:
        hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
        output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)
        # We can use , weights to change the importance of each device
        # example weights = [1, 0.5] -> giving more importance to the first device...
        # is like percentage of importance :  sum(a * weights) / sum(weights)

    #################
    # Sending models
    #################
    for d in devices_connected:
        print(f'Sending model to {d.port} ...')

        ini_time = time.time()
        for i in range((650+1)*16): # hidden layer
            d.read() # wait until confirmatio
            float_num = hidden_layer[i]
            data = struct.pack('f', float_num)
            d.write(data)

        for i in range((16+1)*3): # output layer
            d.read() # wait until confirmatio
            float_num = output_layer[i]
            data = struct.pack('f', float_num)
            d.write(data)

        print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')

###########
# END MAIN
###########




# if __name__ == "__main__":
#     main()