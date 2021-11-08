# Federated Learning with Arduino Nano 33 BLE Sense

Final degree project for the Bachelor's Degree in Computer Science of the [Universitat Politècnica de Catalunya](https://www.upc.edu/ca) - [Facultat d'informàtica de Barcelona](https://www.fib.upc.edu/).


## How tu use it
1. Configure the Arduino Nano 33 BLE Sense boards like in the image. ![board setup](images/arduino_sketch.png)
2. Open the project with PlatformIO and flash the firmware to all the boards.
3. Run the fl_server.py using Python3
    1. Specify the number of devices used
    2. Specify the Serial ports of each device
4. Start training the devices using the buttons.
    * The 3 buttons on the left are used to train 3 different keywords (to be decided by you!)
    * The board will start recording when the button is pressed & RELEASED (one second).
    * The fourth buttons is used to start the Federated Learning process.

## Authors
- Marc Monfort
- Nil Llisterri