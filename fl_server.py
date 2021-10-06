import serial
from serial.tools.list_ports import comports

import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time


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
    colors = ['r', 'g', 'b', 'y', 'p']
    devices =  [x[2] for x in graph_data]
    for device_index in devices:
        epoch = [x[0] for x in graph_data if x[2] == device_index]
        error = [x[1] for x in graph_data if x[2] == device_index]
    
        plt.plot(error, colors[device_index], label=device_index)
    plt.xlim(xmin=0.0) 
    plt.ylim(ymin=0.0)
    plt.ylabel('Loss') # or Error
    plt.xlabel('Epoch')
   
    plt.show()
    plt.autoscale()
    plt.pause(0.1)


pauseListen = False # So there are no threads reading the serial input at the same time
def listenDevice(device, deviceIndex):
    global pauseListen, graph
    while True:
        while (pauseListen):
            print("Paused...")
            time.sleep(0.1)

        d.timeout = None
        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            # Modified to graph
            if msg[:-2] == 'graph':
                ne = device.readline()[:-2];
                n_epooch = int(ne)

                n_error = device.read(4)
                [n_error] = struct.unpack('f', n_error)
                nb = device.readline()[:-2]
                graph.append([n_epooch, n_error, deviceIndex])

            elif msg[:-2] == 'start_fl':
                startFL()


def getDevices():
    global devices
    num_devices = read_number("Number of devices: ")

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports:
        print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]


def getModel(d, device_index, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected):
    global size_hidden_layer, size_output_layer
    d.reset_input_buffer()
    d.reset_output_buffer()
    d.timeout = 5

    print(f'Starting connection to {d.port} ...') # Hanshake
    d.write(b'>') # Python --> SYN --> Arduino
    if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
        d.write(b's') # Python --> ACK --> Arduino
        
        print('Connection accepted.')
        devices_connected.append(d)
        #devices_hidden_layer = np.vstack((devices_hidden_layer, np.empty(size_hidden_layer)))
        #devices_output_layer = np.vstack((devices_output_layer, np.empty(size_output_layer)))
        d.timeout = None

        print_until_keyword('start', d)
        devices_num_epochs.append(int(d.readline()[:-2]))

        print(f'Receiving model from {d.port} ...')
        ini_time = time.time()

        for i in range(size_hidden_layer): # hidden layer
            data = d.read(4)
            [float_num] = struct.unpack('f', data)
            devices_hidden_layer[device_index][i] = float_num

        for i in range(size_output_layer): # output layer
            data = d.read(4)
            [float_num] = struct.unpack('f', data)
            devices_output_layer[device_index][i] = float_num

        print(f'Model received from {d.port} ({time.time()-ini_time} seconds)')

        # if it was not connected before, we dont use the devices' model
        if not d in old_devices_connected:
            devices_num_epochs[device_index] = 0
            print(f'Model not used. The device {d.port} has an outdated model')

    else:
        print(f'Connection timed out. Skipping {d.port}.')

def sendModel(d, hidden_layer, output_layer):
    ini_time = time.time()
    for i in range(size_hidden_layer): # hidden layer
        d.read() # wait until confirmatio
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        d.write(data)

    for i in range(size_output_layer): # output layer
        d.read() # wait until confirmatio
        float_num = output_layer[i]
        data = struct.pack('f', float_num)
        d.write(data)

    print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')

getDevices()

graph = []
size_hidden_nodes = 20
size_hidden_layer = (650+1)*size_hidden_nodes
size_output_layer = (size_hidden_nodes+1)*3

np.random.seed(12345)
hidden_layer = np.random.uniform(-0.5,0.5, size_hidden_layer).astype('float32')
output_layer = np.random.uniform(-0.5, 0.5, size_output_layer).astype('float32')

# To load a Pre-trained model
# hidden_layer = np.load("./hidden_montserrat.npy")
# output_layer = np.load("./output_montserrat.npy")


# Send the blank model to all the devices
threads = []
for d in devices:
    thread = threading.Thread(target=init_network, args=(hidden_layer, output_layer, d))
    thread.daemon = True
    thread.start()
    threads.append(thread)
for thread in threads: thread.join() # Wait for all the threads to end

# Listen their updates
for i, d in enumerate(devices):
    thread = threading.Thread(target=listenDevice, args=(d, i))
    thread.daemon = True
    thread.start()

devices_connected = devices


def startFL():
    global devices_connected, hidden_layer, output_layer, pauseListen

    pauseListen = True

    print('Starting Federated Learning')
    old_devices_connected = devices_connected
    devices_connected = []
    devices_hidden_layer = np.empty((len(devices), size_hidden_layer), dtype='float32')
    devices_output_layer = np.empty((len(devices), size_output_layer), dtype='float32')
    devices_num_epochs = []

    ##################
    # Receiving models
    ##################
    threads = []
    for i, d in enumerate(devices):
        thread = threading.Thread(target=getModel, args=(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end

    
    ####################
    # Processing models
    ####################

    # if sum == 0, any device made any epoch
    if sum(devices_num_epochs) > 0:
        # We can use weights to change the importance of each device
        # example weights = [1, 0.5] -> giving more importance to the first device...
        # is like percentage of importance :  sum(a * weights) / sum(weights)
        hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
        output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)


    #################
    # Sending models
    #################
    threads = []
    for d in devices_connected:
        print(f'Sending model to {d.port} ...')

        thread = threading.Thread(target=sendModel, args=(d, hidden_layer, output_layer))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end


    pauseListen = False

plt.ion()
plt.show()
while True:
    plot_graph(graph)
    time.sleep(0.5)
