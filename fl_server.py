import serial
from serial.tools.list_ports import comports

import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random
from queue import Queue

random.seed(4321)
np.random.seed(4321)

mixedPrecision = True
scaledWeightsSize = 1
samples_per_device = 120 # Amount of samples of each word to send to each device
batch_size = 15 # Must be even, hsa to be split into 2 types of samples
experiment = 'iid' # 'iid', 'no-iid', 'train-test', None
use_threads = True

debug = True

test_samples_amount = 60
size_hidden_nodes = 25
size_hidden_layer = (650+1)*size_hidden_nodes
size_output_layer = (size_hidden_nodes+1)*3
momentum = 0.9
learningRate= 0.6
pauseListen = False # So there are no threads reading the serial input at the same time

montserrat_files = [file for file in os.listdir("datasets/mountains") if file.startswith("montserrat")]
pedraforca_files = [file for file in os.listdir("datasets/mountains") if file.startswith("pedraforca")]
vermell_files = [file for file in os.listdir("datasets/colors") if file.startswith("vermell")]
verd_files = [file for file in os.listdir("datasets/colors") if file.startswith("verd")]
blau_files = [file for file in os.listdir("datasets/colors") if file.startswith("blau")]
test_montserrat_files = [file for file in os.listdir("datasets/test/") if file.startswith("montserrat")]
test_pedraforca_files = [file for file in os.listdir("datasets/test") if file.startswith("pedraforca")]

random.shuffle(montserrat_files)
random.shuffle(pedraforca_files)

mountains = list(sum(zip(montserrat_files, pedraforca_files), ()))

test_mountains = list(sum(zip(test_montserrat_files, test_pedraforca_files), ()))


def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword:
            break
        else:
            print(f'({arduino.port}):',msg, end='')

def init_network(hidden_layer, output_layer, device, deviceIndex):
    device.reset_input_buffer()
    device.write(b's')
    print_until_keyword('start', device)
    print(f"Sending model to {device.port}")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', momentum))

    for i in range(len(hidden_layer)):
        if i < 5 and d.port == 'com6': print(f"[{d.port}] Init Weight {i}: {hidden_layer[i]}")
        device.read() # wait until confirmation of float received
        data = struct.pack('f', hidden_layer[i])
        device.write(data)
    
    for i in range(len(output_layer)):
        device.read() # wait until confirmation of float received
        data = struct.pack('f', output_layer[i])
        device.write(data)

    print(f"Model sent to {device.port}")
    modelReceivedConfirmation = device.readline().decode()
    # print(f"Model received confirmation: ", modelReceivedConfirmation)
    
# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, mountains

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size

    if debug: print(f"[{device.port}] Sending samples from {start} to {end}")

    files = mountains[start:end]
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        if debug: print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex)

def sendSamplesNonIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, vermell_files, verd_files, blau_files

    start = (batch_index * batch_size)
    end = (batch_index * batch_size) + batch_size

    dir = 'datasets/' # TMP fix
    if (deviceIndex == 0):
        files = vermell_files[start:end]
        num_button = 1
        dir = 'colors'
    elif  (deviceIndex == 1):
        files = montserrat_files[start:end]
        num_button = 2
        dir = 'mountains'
    elif  (deviceIndex == 2):
        files = pedraforca_files[start:end]
        num_button = 3
        dir = 'mountains'
    else:
        exit("Exceeded device index")

    for i, filename in enumerate(files):
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        sendSample(device, f"datasets/{dir}/{filename}", num_button, deviceIndex)

def sendSample(device, samplePath, num_button, deviceIndex, only_forward = False):
    with open(samplePath) as f:
        ini_time = time.time() * 1000
        data = json.load(f)
        device.write(b't')
        startConfirmation = device.readline().decode()
        if debug: print(f"[{device.port}] Train start confirmation:", startConfirmation)

        device.write(struct.pack('B', num_button))
        button_confirmation = device.readline().decode() # Button confirmation
        if debug: print(f"[{device.port}] Button confirmation: {button_confirmation}")

        device.write(struct.pack('B', 1 if only_forward else 0))
        only_forward_conf = device.readline().decode()
        if debug: print(f"[{device.port}] Only forward confirmation: {only_forward_conf}") # Button confirmation

        for i, value in enumerate(data['payload']['values']):
            device.write(struct.pack('h', value))

        sample_received_conf = device.readline().decode()
        if debug: print(f"[{device.port}] Sample received confirmation:", sample_received_conf)

        # print(f"Fordward millis received: ", device.readline().decode())
        # print(f"Backward millis received: ", device.readline().decode())
        device.readline().decode() # Accept 'graph' command
        error, num_button_predicted = read_graph(device, deviceIndex)
        if (debug and error > 0.28):
            print(f"[{device.port}] Sample {samplePath} generated an error of {error}")
        print(f'[{device.port}] Sample sent in: {(time.time()*1000)-ini_time} milliseconds)')
    
    # print(f"{num_button} - {num_button_predicted}")
    return error, num_button == num_button_predicted

def sendTestSamples(device, deviceIndex, successes_queue):
    global test_mountains

    # print(f"[{device.port}] Sending test samples from {0} to {60}")

    start = deviceIndex*test_samples_amount
    end = (deviceIndex*test_samples_amount) + test_samples_amount
   
    files = test_mountains[start:end]
   
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        elif (filename.startswith("vermell")):
            num_button = 3
        else:
            exit("Unknown button for sample")
        
        error, success = sendSample(device, 'datasets/test/'+filename, num_button, deviceIndex, True)
        successes_queue.put(success)

def read_graph(device, deviceIndex):
    global repaint_graph

    outputs = device.readline().decode().split()
    if debug: print(f'Ouptuts: {outputs}')

    bpred = outputs.index(max(outputs))+1
    if debug: print(f'Predicted button: {bpred}')
    
    error = device.readline().decode()
    if debug: print(f"Error: ", error)

    ne = device.readline()[:-2]
    n_epooch = int(ne)

    n_error = device.read(4)
    [n_error] = struct.unpack('f', n_error)
    nb = device.readline()[:-2]
    graph.append([n_epooch, n_error, deviceIndex])
    repaint_graph = True
    return n_error, outputs.index(max(outputs)) + 1

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
            #port = "COM8";
            return serial.Serial(port, 9600)
        except:
            print(f"ERROR: Wrong port connection ({port})")

first_paint = True
graph = []
repaint_graph = True
def plot_graph():
    global graph, repaint_graph, devices, first_paint

    if (repaint_graph):
        colors = ['r', 'g', 'b', 'y']
        markers = ['-', '--', ':', '-.']
        #devices =  [x[2] for x in graph]
        epochs = 1
        for device_index, device in enumerate(devices):
            epoch = [x[0] for x in graph if x[2] == device_index]
            error = [x[1] for x in graph if x[2] == device_index]

            epochs = max(len(error), epochs)
        
            plt.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}")

        if (first_paint):
            plt.legend()
            plt.xlim(left=0)
            plt.ylim(bottom=0, top=0.7)
            plt.ylabel('Loss') # or Error
            plt.xlabel('Epoch')
            first_paint = False
        # plt.axes().set_ylim([0, 0.6])
        # plt.xlim(bottom=0)
        plt.autoscale(axis='x')
        #plt.xlim = epochs
        #plt.xticks(range(0, epochs))

        if (experiment == 'train-test'):
            plt.axvline(x=samples_per_device)

        repaint_graph = False

    plt.pause(2)

    

def listenDevice(device, deviceIndex):
    global pauseListen, graph
    while True:
        while (pauseListen):
            time.sleep(0.1)

        d.timeout = None
        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            # Modified to graph
            if msg[:-2] == 'graph':
                read_graph(device, deviceIndex)

            elif msg[:-2] == 'start_fl':
                startFL()

def getDevices():
    global devices, devices_connected
    num_devices = read_number("Number of devices: ")
    # num_devices = 2

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports:
        print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]
    devices_connected = devices

def FlGetModel(d, device_index, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected):
    global size_hidden_layer, size_output_layer
    #d.reset_input_buffer()
    #d.reset_output_buffer()
    #d.timeout = 5

    print(f'[{d.port}] Starting connection...') # Handshake
    d.write(b'>') # Python --> SYN --> Arduino
    syn_ack = d.read()
    print(f'[{d.port}] syn_ack: {syn_ack}') # Handshake

    # if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
    d.write(b's') # Python --> ACK --> Arduino
    
    print(f"[{d.port}] Connection accepted.")
    devices_connected.append(d)
    d.timeout = None

    print_until_keyword('start', d)
    num_epochs = int(d.readline()[:-2])
    devices_num_epochs.append(num_epochs)
    print(f"[{d.port}] Num epochs: {num_epochs}")


    min_w = readFloat(d)
    max_w = readFloat(d)

    # print(f"[{d.port}] Scaled weight size: {scaledWeightsSize}")
    print(f"[{d.port}] Min weight: {min_w}, max weight: {max_w}")
    a, b = getScaleRange()
    print(f"[{d.port}] Scaling precision: {(abs(max_w-min_w)) / abs(a-b)}")


    print(f"[{d.port}] Receiving model...")
    ini_time = time.time()

    for i in range(size_hidden_layer): # hidden layer
        if mixedPrecision:
            scaledWeight = readInt(d, scaledWeightsSize)
            #scaled_in_float = readFloat(d)
            # float_weight = readFloat(d)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            # if i < 5 and d.port == 'com6': print(f"[{d.port}] Recevied Weight {i}: {float_weight}")
            # if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else: 
            weight = readFloat(d)
        
        devices_hidden_layer[device_index][i] = weight

    for i in range(size_output_layer): # output layer
        if mixedPrecision:
            scaledWeight = readInt(d, scaledWeightsSize)
            #scaled_in_float = readFloat(d)
            # float_weight = readFloat(d)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            #if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else:
            weight = readFloat(d)
        
        devices_output_layer[device_index][i] = weight

    print(f"[{d.port}] Model received from {d.port} ({time.time()-ini_time} seconds)")

    # if it was not connected before, we dont use the devices' model
    if not d in old_devices_connected:
        devices_num_epochs[device_index] = 0
        print(f"[{d.port}] Model not used. The device has an outdated model")

    #else:
    #    print(f"[{d.port}] Connection timed out, skipping.")

def readFloat(d):
    data = d.read(4)
    [float_num] = struct.unpack('f', data)
    return float_num

def readInt(d, size):
    return int.from_bytes(d.read(size), "little", signed=True)

def deScaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return min_w + ( (weight-a)*(max_w-min_w) / (b-a) )

def scaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return int(a + ( (weight-min_w)*(b-a) / (max_w-min_w) ))


def getScaleRange():
    if scaledWeightsSize == 1:
        return -128, 127
    elif scaledWeightsSize == 2:
        return -32768, 32767
    elif scaledWeightsSize == 4:
        return -2147483648, 2147483647

def sendModel(d, hidden_layer, output_layer):

    min_w = min(min(hidden_layer), min(output_layer))
    max_w = max(max(hidden_layer), max(output_layer))
    print(f"Min weight to send: {min_w}, max: {max_w}")
    
    d.write(struct.pack('f', min_w))
    d.write(struct.pack('f', max_w))

    ini_time = time.time()
    for i in range(size_hidden_layer): # hidden layer
        if i < 5 and d.port == 'com6': print(f"[{d.port}] Sending weight {i}: {hidden_layer[i]}")
        #d.read() # wait until confirmatio
        if mixedPrecision:
            scaled = scaleWeight(min_w, max_w, hidden_layer[i])
            d.write(scaled.to_bytes(scaledWeightsSize, "little", signed=True))
        else:
            float_num = hidden_layer[i]
            d.write(struct.pack('f', float_num))

    for i in range(size_output_layer): # output layer
        #d.read() # wait until confirmatio
        if mixedPrecision:
            scaled = scaleWeight(min_w, max_w, output_layer[i])
            d.write(scaled.to_bytes(scaledWeightsSize, "little", signed=True))
        else:
            float_num = output_layer[i]
            d.write(struct.pack('f', float_num))

    print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')

def startFL():
    global devices_connected, hidden_layer, output_layer, pauseListen

    pauseListen = True

    print('Starting Federated Learning')
    old_devices_connected = devices_connected
    devices_connected = []
    devices_hidden_layer = np.empty((len(devices), size_hidden_layer), dtype='float32')
    devices_output_layer = np.empty((len(devices), size_output_layer), dtype='float32')
    devices_num_epochs = []

    # Receiving models
    threads = []
    for i, d in enumerate(devices):
        if use_threads:
            thread = threading.Thread(target=FlGetModel, args=(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            FlGetModel(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected)
 
    for thread in threads: thread.join() # Wait for all the threads to end

    # Processing models

    # if sum == 0, any device made any epoch
    print(f"Devices num epochs: {devices_num_epochs}")
    if sum(devices_num_epochs) > 0:
        # We can use weights to change the importance of each device
        # example weights = [1, 0.5] -> giving more importance to the first device...
        # is like percentage of importance :  sum(a * weights) / sum(weights)
        ini_time = time.time() * 1000
        hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
        output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)
        print(f'Average millis: {(time.time()*1000)-ini_time} milliseconds)')


    # Sending models
    threads = []
    for d in devices_connected:
        print(f'Sending model to {d.port} ...')

        if use_threads:
            thread = threading.Thread(target=sendModel, args=(d, hidden_layer, output_layer))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            sendModel(d, hidden_layer, output_layer)
    for thread in threads: thread.join() # Wait for all the threads to end
    pauseListen = False


getDevices()

# To load a Pre-trained model
# hidden_layer = np.load("./hidden_montserrat.npy")
# output_layer = np.load("./output_montserrat.npy")


# Send the blank model to all the devices
threads = []
for i, d in enumerate(devices):
    hidden_layer = np.random.uniform(-0.5,0.5, size_hidden_layer).astype('float32')
    output_layer = np.random.uniform(-0.5, 0.5, size_output_layer).astype('float32')

    if use_threads:
        thread = threading.Thread(target=init_network, args=(hidden_layer, output_layer, d, i))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    else:
        init_network(hidden_layer, output_layer, d, i)
for thread in threads: thread.join() # Wait for all the threads to end

if experiment != None:
    train_ini_time = time.time()
    # Train the device
    for batch in range(int(samples_per_device/batch_size)):
        batch_ini_time = time.time()
        for deviceIndex, device in enumerate(devices):
            if experiment == 'iid' or experiment == 'train-test':
                method = sendSamplesIID
            elif experiment == 'no-iid':
                method = sendSamplesNonIID

            if use_threads:
                thread = threading.Thread(target=method, args=(device, deviceIndex, batch_size, batch))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            else:
                method(device, deviceIndex, batch_size, batch)
        for thread in threads: thread.join() # Wait for all the threads to end
        print(f'Batch time: {time.time() - batch_ini_time} seconds)')
        fl_ini_time = time.time()
        startFL()
        print(f'FL time: {time.time() - fl_ini_time} seconds)')
        time.sleep(1)

    train_time = time.time()-train_ini_time
    # print(f'Trained in ({train_time} seconds)')

    if experiment == 'train-test':
        successes_queue = Queue()
        for deviceIndex, device in enumerate(devices):
            if use_threads:
                thread = threading.Thread(target=sendTestSamples, args=(device, deviceIndex, successes_queue))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            else:
                sendTestSamples(device, deviceIndex, successes_queue)
        for thread in threads: thread.join() # Wait for all the threads to end

        test_accuracy = sum(successes_queue.queue)/len(successes_queue.queue)
        print(f"Testing accuracy: {test_accuracy}")
        print(f"{test_accuracy}, ", end = '')


plt.ion()
# plt.title(f"Loss vs Epoch")
plt.show()

font_sm = 13
font_md = 16
font_xl = 18
plt.rc('font', size=font_sm)          # controls default text sizes
plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_sm)    # legend fontsize
plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title

plot_graph()

if experiment != None:
    figname = f"plots/BS{batch_size}-LR{learningRate}-M{momentum}-HL{size_hidden_nodes}-TT{train_time}-{experiment}.png"
    plt.savefig(figname, format='png')
    print(f"Generated {figname}")

# Listen their updates
for i, d in enumerate(devices):
    thread = threading.Thread(target=listenDevice, args=(d, i))
    thread.daemon = True
    thread.start()

while True:
    #if (repaint_graph): 
    plot_graph()
        #repaint_graph = False
    # time.sleep(0.1)
