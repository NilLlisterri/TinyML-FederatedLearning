# Federated Learning with Arduino Nano/Portenta

## How tu run it
1. Open the project with PlatformIO and flash the firmware to all the boards.  
`pio run --target upload -e portenta_h7_m7 --upload-port com4`
2. Run the fl_server.py using Python3: 
    1. Execute `python fl_server.py`
    2. Specify the number of devices used
    3. Specify the Serial ports of each device
    4. [Optional] Change the NN parameters
