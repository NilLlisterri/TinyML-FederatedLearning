from ast import keyword
import warnings
from matplotlib import figure
from matplotlib.ticker import MaxNLocator
import serial
from serial.tools.list_ports import comports
import struct
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random
from queue import Queue
from scipy.io import wavfile
import sys
# import tensorflow as tf

seed = 4789 # 4789
random.seed(seed)
np.random.seed(seed)

# Keyword samples split
samples_folder = "./datasets/keywords_v3"
train_samples_split = 160 # Number of samples for training of each keyword
test_samples_split = 20   # Number of samples for training of each keyword

# Experiment sizes
training_epochs = 160     # Amount of training epochs. Can't be more than kws * train_samples_split
testing_epochs = 60       # Amount of test samples of each keyword. Can't be more than kws * test_samples_split

# Federated learning
enableFL = True
enableTest = True
enablePlot = False
batch_size = 4 #90        # Must be divisble by the amount of keywords
mixedPrecision = True
scaledWeightsBytes = 2
scaledWeightBits = 16
interactive = True

experiment = 'iid'        # 'iid', 'no-iid', 'train-test', None
debug = True
pauseListen = False       # So there are no threads reading the serial input at the same time

# NN Parameters
output_nodes = 4          # len(keywords_buttons)
size_hidden_nodes = 50
size_hidden_layer = (650+1)*size_hidden_nodes
size_output_layer = (size_hidden_nodes+1)*output_nodes
momentum = 0.9
learningRate= 0.05

keywords_buttons = {
    "montserrat": 1,
    "pedraforca": 2,
    "vermell": 3,
    "blau": 4,
}

def WAIT_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword: break
        # else: print(f'({arduino.port}):',msg, end='')

def initDevice(hidden_layer, output_layer, device):
    device.reset_input_buffer()
    device.write(b's')
    WAIT_until_keyword('start', device)
    device.write(struct.pack('i', seed))
    # print(f"Seed conf: {device.readline().decode()}")
    if debug: print(f"[{device.port}] Sending blank model...")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', momentum))

    for i in range(len(hidden_layer)):
        device.read() # wait until confirmation of float received
        device.write(struct.pack('f', hidden_layer[i]))
    
    for i in range(len(output_layer)):
        device.read() # wait until confirmation of float received
        device.write(struct.pack('f', output_layer[i]))

    if debug: print(f"[{device.port}] Model sent")
    modelReceivedConfirmation = device.readline().decode()
    # if debug: print(f"[{device.port}] Model received confirmation: ", modelReceivedConfirmation)
    
# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index):
    start = ((deviceIndex*training_epochs) + (batch_index * batch_size)) #%len(keywords)
    end = ((deviceIndex*training_epochs) + (batch_index * batch_size) + batch_size) #%len(keywords)

    if debug: print(f"[{device.port}] Sending samples of batch {batch_index + 1}, from {start+1} to {end}")

    for i in range(start, end):
        filename = keywords[i % len(keywords)]
        keyword = filename.split("/")[0]
        num_button = keywords_buttons[keyword]

        if debug: print(f"[{device.port}] Sending sample {i+1} ({i+1-start}/{end-start})")
        error, success = sendSample(device, f"{samples_folder}/{filename}", num_button, deviceIndex) # 'datasets/keywords/'
        successes_map[deviceIndex].put(success)
        training_errors_map[deviceIndex].append(error)
        samplesAccuracyTick = sum(successes_map[deviceIndex].queue)/len(successes_map[deviceIndex].queue)
        # if debug: print(f"[{device.port}] Samples accuracy tick: {samplesAccuracyTick}")
        training_accuracy_map[deviceIndex].append(samplesAccuracyTick)

def sendSamplesNonIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, blau_files, verd_files, vermell_files

    start = (batch_index * batch_size)
    end = (batch_index * batch_size) + batch_size

    if (deviceIndex == 0):
        files = vermell_files[start:end]
        num_button = 1
    elif  (deviceIndex == 1):
        files = montserrat_files[start:end]
        num_button = 2
    elif  (deviceIndex == 2):
        files = pedraforca_files[start:end]
        num_button = 3
    else:
        exit("Exceeded device index")

    for i, filename in enumerate(files):
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        error, success = sendSample(device, f"datasets/keywords/{filename}", num_button, deviceIndex)
        successes_map[deviceIndex].put(success)
        training_errors_map[deviceIndex].put(error)
        training_accuracy_map[deviceIndex].append(sum(successes_map[deviceIndex].queue)/len(training_errors_map[deviceIndex].queue))

def sendSample(device, samplePath, num_button, deviceIndex, only_forward = False):
    with open(samplePath) as f:
        ini_time = time.time() * 1000
        # if debug: print(f'[{device.port}] Sending train command')
        device.write(b't')
        startConfirmation = device.readline().decode()
        #if debug: print(f"[{device.port}] Train start confirmation:", startConfirmation)

        device.write(struct.pack('B', num_button))
        button_confirmation = device.readline().decode() # Button confirmation
        # if debug: print(f"[{device.port}] Button confirmation: {button_confirmation}")

        device.write(struct.pack('B', 1 if only_forward else 0))
        only_forward_conf = device.readline().decode()
        # if debug: print(f"[{device.port}] Only forward confirmation: {only_forward_conf}") # Button confirmation

        if samplePath.endswith('.wav'):
            samplerate, values = wavfile.read(f"{samples_folder}/{keywords[0]}")
        elif samplePath.endswith('.json'):
            data = json.load(f)
            values = data['payload']['values']
        
        for value in values:
            device.write(struct.pack('h', value))
            # device.read()

        sample_received_conf = device.readline().decode()
        # if debug: print(f"[{device.port}] Sample received confirmation:", sample_received_conf)

        graphCommand = device.readline().decode()
        # if debug: print(f"[{device.port}] Graph command received: {graphCommand}")
        error, num_button_predicted = read_graph(device, deviceIndex)
        # if debug: print(f'[{device.port}] Sample sent in: {(time.time()*1000)-ini_time} milliseconds)')

    return error, num_button == num_button_predicted

def sendTestSamples(device, deviceIndex):
    global test_keywords, test_accuracies_map, test_errors_map

    errors_queue = Queue()
    successes_queue = Queue()

    if debug: print(f"[{device.port}] Sending {testing_epochs} test samples")
    for i, filename in enumerate(test_keywords[:testing_epochs]):
        if debug: print(f"[{device.port}] Sending test sample {i}/{testing_epochs}")
        keyword = filename.split("/")[0]
        num_button = keywords_buttons[keyword]
        
        error, success = sendSample(device, 'datasets/keywords_v3/'+filename, num_button, deviceIndex, True)
        errors_queue.put(error)
        successes_queue.put(success)
    
    test_accuracy = sum(successes_queue.queue)/len(successes_queue.queue)
    test_error = sum(errors_queue.queue)/len(errors_queue.queue)
    print(f"[{device.port}] Testing accuracy: {test_accuracy}")
    print(f"[{device.port}] Testing MSE: {test_error}")
    test_accuracies_map[deviceIndex].append(test_accuracy)
    test_errors_map[deviceIndex].append(test_error)

def sendTestAllDevices():
    global devices
    threads = []
    for deviceIndex, device in enumerate(devices):
        thread = threading.Thread(target=sendTestSamples, args=(device, deviceIndex))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join()

def read_graph(device, deviceIndex):
    outputs = device.readline().decode().split()
    # print(f'[{device.port}] Outputs: {outputs}')

    predicted_button = outputs.index(max(outputs))+1
    # if debug: print(f'[{device.port}] Predicted button: {predicted_button}')
    
    error = device.readline().decode()
    # print(f"[{device.port}] Error: {error}")

    ne = device.readline()[:-2]
    n_epooch = int(ne)

    n_error = device.read(4)
    [n_error] = struct.unpack('f', n_error)
    nb = device.readline()[:-2]
    graph.append([n_epooch, n_error, deviceIndex])
    return n_error, outputs.index(max(outputs)) + 1

def read_number_devices(msg):
    while True:
        try:
            return int(input(msg))
        except: print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            return serial.Serial(port, 9600)
        except: print(f"ERROR: Wrong port connection ({port})")

def listenDevice(device, deviceIndex):
    global pauseListen, graph
    while True:
        while (pauseListen): time.sleep(0.1)

        d.timeout = None
        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            if msg[:-2] == 'graph': read_graph(device, deviceIndex)
            elif msg[:-2] == 'start_fl': startFL()

def getDevices():
    global learningRate, momentum

    if interactive: 
        num_devices = read_number_devices("Number of devices: ")
        print("Available ports: ")
        for available_port in comports(): print(f"\t{available_port}")
        devices = [read_port(f"Port for device {i+1}: ") for i in range(num_devices)]
        learningRate = float(input(f"Learning rate? [{learningRate}]: ") or learningRate)
        momentum = float(input(f"Momentum? [{momentum}]: ") or momentum)
    else:
        num_devices = 3
        devices = [serial.Serial("com4", 9600), serial.Serial("com8", 9600), serial.Serial("com14", 9600)]
    return devices

def getModel(d, device_index, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected):
    global size_hidden_layer, size_output_layer

    # if debug: print(f'[{d.port}] Starting connection...') # Handshake
    d.write(b'>') # Python --> SYN --> Arduino
    syn_ack = d.read()
    # if debug: print(f'[{d.port}] syn_ack: {syn_ack}') # Handshake

    # if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
    d.write(b's') # Python --> ACK --> Arduino
    
    # if debug: print(f"[{d.port}] Connection accepted.")
    devices_connected.append(d)
    d.timeout = None

    WAIT_until_keyword('start', d)
    num_epochs = int(d.readline()[:-2])
    devices_num_epochs.append(num_epochs)
    print(f"[{d.port}] New epochs: {num_epochs}")

    min_w = readFloat(d)
    max_w = readFloat(d)

    min_weights_map[device_index].append(min_w)
    max_weights_map[device_index].append(max_w)

    # print(f"[{d.port}] Scaled weight size: {scaledWeightsBytes}")
    if debug: print(f"[{d.port}] Min weight: {round(min_w, 5)}, max weight: {round(max_w, 5)}")
    a, b = getScaleRange()
    if debug: print(f"[{d.port}] Quantization scale range: {a} - {b}")
    if debug: print(f"[{d.port}] Quantization precision: {round((abs(max_w-min_w)) / abs(a-b), 5)}")

    # if debug: print(f"[{d.port}] Receiving model...")
    ini_time = time.time()

    scaledWeights = []
    unscaledWeights = []
    unmixedWeights = []

    for i in range(size_hidden_layer): # hidden layer
        if mixedPrecision:
            scaledWeight = readInt(d, scaledWeightsBytes)
            # if d.port == "com4":
            #     print(scaledWeight)
            scaledWeights.append(scaledWeight)
            # if i % 100 == 0 and debug: print(f"[{d.port}] Received scaled weight: {scaledWeight}")
            #scaled_in_float = readFloat(d)
            # float_weight = readFloat(d)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            unscaledWeights.append(weight)
            # if i % 100 == 0 and debug: print(f"[{d.port}] Recevied unscaled weight {i}: {weight:.5f}")
            # if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else: weight = readFloat(d)
        unmixedWeights.append(weight)
        # if debug and i % 100 == 0: print(f"[{d.port}] Received Weight {i}: {weight}")
        devices_hidden_layer[device_index][i] = weight
    
    if debug: print(f"[{d.port}] Received hidden layer weights")

    for i in range(size_output_layer): # output layer
        if mixedPrecision:
            scaledWeight = readInt(d, scaledWeightsBytes)
            scaledWeights.append(scaledWeight)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            unscaledWeights.append(weight)
            #if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else: weight = readFloat(d)
        unmixedWeights.append(weight)
        # if debug and i % 100 == 0: print(f"[{d.port}] Received Weight {i}: {weight}")
        devices_output_layer[device_index][i] = weight

    print(f"[{d.port}] Model received in {round(time.time()-ini_time, 3)}s")
    #if d.port == "com14":
        #print("Weight distribution:")
        #np.set_printoptions(threshold=sys.maxsize)
        #print(np.array(np.unique(scaledWeights, return_counts=True)).T)
        #print(np.array(unmixedWeights))

    # if it was not connected before, we dont use the devices' model
    if not d in old_devices_connected:
        devices_num_epochs[device_index] = 0
        print(f"[{d.port}] Model not used. The device has an outdated model")

def readFloat(d):
    data = d.read(4)
    [float_num] = struct.unpack('f', data)
    return float_num

def readInt(d, size):
    return int.from_bytes(d.read(size), "little", signed=False)

def deScaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return min_w + ( (weight-a)*(max_w-min_w) / (b-a) )

def scaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return round(a + ( (weight-min_w)*(b-a) / (max_w-min_w) ))

def getScaleRange():
    return 0, pow(2, scaledWeightBits)-1

#min_w_progress = []
#max_w_progress = []
def sendModel(d, hidden_layer, output_layer):
    if debug: print(f'[{d.port}] Sending the new model...')

    min_w = min(min(hidden_layer), min(output_layer))
    max_w = max(max(hidden_layer), max(output_layer))

    #if d.port == "com4":
    #    min_w_progress.append(min_w)
    #    max_w_progress.append(max_w)
    # if debug: print(f"[{d.port}] Min weight to send: {min_w}, max: {max_w}")
    
    d.write(struct.pack('f', min_w))
    d.write(struct.pack('f', max_w))

    if debug: ini_time = time.time()
    for i in range(size_hidden_layer): # hidden layer
        # if debug and i < 5: print(f"[{d.port}] Sending weight {i}: {hidden_layer[i]}")
        if mixedPrecision:
            scaled = scaleWeight(min_w, max_w, hidden_layer[i])
            d.write(scaled.to_bytes(scaledWeightsBytes, "little", signed=False))
        else:
            float_num = hidden_layer[i]
            d.write(struct.pack('f', float_num))

    for i in range(size_output_layer): # output layer
        if mixedPrecision:
            scaled = scaleWeight(min_w, max_w, output_layer[i])
            d.write(scaled.to_bytes(scaledWeightsBytes, "little", signed=False))
        else:
            float_num = output_layer[i]
            d.write(struct.pack('f', float_num))

    confirmation = d.readline().decode()
    if debug: print(f'[{d.port}] Model sent in {time.time()-ini_time}s')
    # if debug: print(f'Model received confirmation: {confirmation}')

def startFL():
    global devices_connected, hidden_layer, output_layer, pauseListen

    fl_ini_time = time.time()
    fl_round_epochs.append((batch+1) * batch_size)
            
    pauseListen = True

    if debug: print('[MAIN] Starting Federated Learning...')
    old_devices_connected = devices_connected
    devices_connected = []
    devices_hidden_layer = np.empty((len(devices), size_hidden_layer), dtype='float32')
    devices_output_layer = np.empty((len(devices), size_output_layer), dtype='float32')
    devices_num_epochs = []

    # Receiving models
    threads = []
    for i, d in enumerate(devices):
        thread = threading.Thread(target=getModel, args=(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected))
        thread.daemon = True
        thread.start()
        threads.append(thread)
 
    for thread in threads: thread.join() # Wait for all the threads to end

    # Processing models
    if sum(devices_num_epochs) > 0:
        # We can use weights to change the importance of each device
        # example weights = [1, 0.5] -> giving more importance to the first device...
        # is like percentage of importance :  sum(a * weights) / sum(weights)
        ini_time = time.time()
        hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
        output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)
        if debug: print(f'[{d.port}] Model merged in {round((time.time())-ini_time, 5)}s')

    # Sending models
    threads = []
    for d in devices_connected:
        thread = threading.Thread(target=sendModel, args=(d, hidden_layer, output_layer))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end
    pauseListen = False

    if debug: print(f"[MAIN] Federated learning ended in {round(time.time() - fl_ini_time, 3)}s")

# Send the blank model to all the devices
def initializeDevices():
    threads = []
    for i, d in enumerate(devices):
        # Xavier / Glorot initializer
        # initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        # hidden_layer = initializer(shape=(1, size_hidden_layer))[0]
        # output_layer = initializer(shape=(1, size_output_layer))[0]

        hidden_layer = np.random.uniform(-0.5,0.5, size_hidden_layer).astype('float32')
        output_layer = np.random.uniform(-0.5, 0.5, size_output_layer).astype('float32')

        # nw = []
        # for w in hidden_layer:
        #     nw.append(scaleWeight(min(hidden_layer), max(hidden_layer), w))
        # print(np.array(np.unique(nw, return_counts=True)).T)
        # plt.plot(nw)
        # plt.show()

        thread = threading.Thread(target=initDevice, args=(hidden_layer, output_layer, d))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end

graph = []
fl_round_epochs = []
def plot(title):
    global graph, devices, old_graph_size, fl_round_epochs

    warnings.filterwarnings("ignore")
    ax = plt.figure(figsize=(11, 5)).gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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

    colors = ['r', 'g', 'b', 'y']
    markers = ['-', '--', ':', '-.']
    
    plt.ion()
    plt.show(block=False)

    while(True):
        plt.clf()

        epochs = 1
        for device_index, device in enumerate(devices):
            epoch = [x[0] for x in graph if x[2] == device_index]
            error = [x[1] for x in graph if x[2] == device_index]
            epochs = max(len(error), epochs)
            plt.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}", marker='o')

        plt.legend()
        plt.xlim(left=0)
        plt.ylim(bottom=0, top=0.8)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title(title)
        plt.autoscale(axis='x')
        # plt.xticks(range(0, epochs))

        if (experiment == 'train-test'): plt.axvline(x=training_epochs)

        for epoch in fl_round_epochs:
            plt.axvline(epoch - 0.5, linestyle = 'dashed')

        plt.pause(0.1)
        time.sleep(0.4)


devices_connected = devices = getDevices()

# Load the dataset
words = list(keywords_buttons.keys())
files = []
test_files = []
for i, word in enumerate(words):
    file_list = os.listdir(f"{samples_folder}/{word}")
    if (len(file_list) < train_samples_split + test_samples_split): 
        sys.exit(f"[MAIN] Not enough samples for keyword {word}")
    random.shuffle(file_list)
    files.append(list(map(lambda f: f"{word}/{f}", file_list[0:train_samples_split])))
    test_files.append(list(map(lambda f: f"{word}/{f}", file_list[train_samples_split:(train_samples_split+test_samples_split)])))

keywords = list(sum(zip(*files), ()))
test_keywords = list(sum(zip(*test_files), ()))

if (training_epochs > len(keywords) / len(devices)):
    sys.exit(f"[MAIN] Not enough training samples for {training_epochs} training epochs on {len(devices)} devices")
if (testing_epochs > len(test_keywords)):
    sys.exit(f"[MAIN] Not enough testing samples for {testing_epochs} testing epochs")

#if debug: print(f"[MAIN] Total available training keywords: {len(keywords)}")
#if debug: print(f"[MAIN] Total available testing keywords: {len(test_keywords)}")

initializeDevices()

test_accuracies_map = {}
test_errors_map = {}
training_accuracy_map = {}
training_errors_map = {}
successes_map = {}          # Booleans

min_weights_map = {}
max_weights_map = {}

for deviceIndex, device in enumerate(devices): 
    training_accuracy_map[deviceIndex] = []
    test_accuracies_map[deviceIndex] = []
    test_errors_map[deviceIndex] = []
    training_errors_map[deviceIndex] = [] # MSE errors
    successes_map[deviceIndex] = Queue() # Amount of right inferences
    min_weights_map[deviceIndex] = []
    max_weights_map[deviceIndex] = []

if enablePlot:
    # Start plotting thread
    thread = threading.Thread(target=plot, args=["MSE Evolution"])
    thread.daemon = True
    thread.start()


if experiment != None:
    train_ini_time = time.time()
    num_batches = int(training_epochs/batch_size)

    # if enableFL: startFL() # So I can get the initial weight distribution plot
    # if enableTest: sendTestAllDevices() # Initial accuracy

    # Train the device
    for batch in range(num_batches):
        batch_ini_time = time.time()
        if debug: print(f"[MAIN] Sending batch {batch + 1}/{num_batches}")
        threads = []
        for deviceIndex, device in enumerate(devices):
            method = sendSamplesIID if experiment == 'iid' or experiment == 'train-test' else sendSamplesNonIID
            thread = threading.Thread(target=method, args=(device, deviceIndex, batch_size, batch))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads: thread.join() # Wait for all the threads to end
        if debug: print(f'[MAIN] Batch time: {round(time.time() - batch_ini_time, 3)}s')
       
        if enableFL: 
            startFL()
        
        time.sleep(1)

        if enableTest:
            if debug: print(f"[{device.port}] Sending test samples")
            sendTestAllDevices() # To calculate the accuracy on every epoch

    if debug:
        print(f'[MAIN] Training completed in {time.time() - train_ini_time}s')

    for deviceIndex, device in enumerate(devices):
        None
        #print(f"[{device.port}] Training loss total: {sum(training_errors_map[deviceIndex])}")
        #print(f"[{device.port}] Training loss mean: {sum(training_errors_map[deviceIndex])/len(training_errors_map[deviceIndex])}")
        #print(f"[{device.port}] Training accuracy: {sum(successes_map[deviceIndex].queue)/len(successes_map[deviceIndex].queue)}")
        #print(f"[{device.port}] Min weight progress: {min_weights_map[deviceIndex]}")
        #print(f"[{device.port}] Max weight progress: {max_weights_map[deviceIndex]}")

if enableTest: sendTestAllDevices()

if experiment == 'train-test': 
    sendTestAllDevices()


# print(f"[MAIN] Training accuracies map: {training_accuracy_map}")
# print(f"[MAIN] Training errors map: {training_errors_map}")
if enableTest:
    print(f"[MAIN] Testing accuracies map: {test_accuracies_map}")
    print(f"[MAIN] Testing errors map: {test_errors_map}")
#print(f"Min weight progress: {min_w_progress}")
#print(f"Max weight progress: {max_w_progress}")



plt.ylim(bottom=0, top=1)
plt.xlim(left=0)
plt.autoscale(axis='x')
for device_index, device in enumerate(devices):
  #plt.plot(training_accuracy_map[device_index], label=f"Device {device_index}")
  plt.plot(test_accuracies_map[device_index], label=f"Device {device_index}")

if experiment != None:
    figname = f"plots/{len(devices)}-{scaledWeightsBytes if mixedPrecision else 'no'}.png"
    plt.savefig(figname, format='png')
    print(f"Generated {figname}")

# plt.show()


time.sleep(10)