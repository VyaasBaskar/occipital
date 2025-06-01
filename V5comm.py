from ctypes import Array
import struct
import threading
from threading import Lock
import json
from json import JSONEncoder
import serial
import time

class Detection:
    def __init__(self, cls: int, tx_l: float, tx_r: float, d: float, h: float, cam_latency: float):
        self.cls = cls
        self.tx_l = tx_l
        self.tx_r = tx_r
        self.d = d
        self.h = h
        self.cam_latency = cam_latency


    def to_Serial(self):
        # Convert Detection properties to serialized binary format
        data = struct.pack('<ifffff', self.cls, self.tx_l, self.tx_r, self.d, self.h, self.cam_latency)
        return data
    
    def to_JSON(self):
        # Convert Detection properties to JSON format
        outData = {}
        outData['cls'] = self.cls
        outData['tx_l'] = self.tx_l
        outData['tx_r'] = self.tx_r
        outData['d'] = self.d
        outData['h'] = self.h
        outData['cam_latency'] = self.cam_latency
        return outData


class AIRecord:
    # The AIRecord is what is communicated from the Jetson to the V5 Brain as a detection
    def __init__(self, detections: "list[Detection]"):
        # Initialize properties of AIRecord class, including position and detections list
        self.detections = detections

    def to_Serial(self):
        # Convert AIRecord properties to serialized binary format
        data = struct.pack('<i', len(self.detections))
        for det in self.detections:
            data += det.to_Serial()
        return data
    
    def to_JSON(self):
        # Convert AIRecord properties to JSON format
        outData = {}
        outData['detections'] = [det.to_JSON() for det in self.detections]
        return outData

    POLYNOMIAL_CRC32 = 0x04C11DB7

    __crc32_table = [0] * 256
    __table32Generated = 0

    def __Crc32GenerateTable(self):
        for i in range(256):
            crc_accum = i << 24
            for j in range(8):
                if crc_accum & 0x80000000:
                    crc_accum = (crc_accum << 1) ^ AIRecord.POLYNOMIAL_CRC32
                else:
                    crc_accum = crc_accum << 1
            AIRecord.__crc32_table[i] = crc_accum
        AIRecord.__table32Generated = 1

    def __Crc32Generate(self, data, accumulator):
        i, j = 0, 0
        if not AIRecord.__table32Generated:
            self.__Crc32GenerateTable()
        for j in range(len(data)):
            i = ((accumulator >> 24) ^ data[j]) & 0xFF
            accumulator = (accumulator << 8) ^ AIRecord.__crc32_table[i]
        return accumulator

    def getCRC32(self):
        data = self.to_Serial()
        crc = self.__Crc32Generate(data, 0) & 0xFFFFFFFF
        return crc
    

class V5SerialPacket:
    def __init__(self, type: int, detections: AIRecord):
        # Initialize properties of V5SerialPacket class, including type and detections
        self.__length = len(detections.to_Serial())
        self.__type = type        # 2 bytes
        self.__detections = detections

    def to_Serial(self):
        # Convert V5SerialPacket properties to serialized binary format
        data = bytearray([0xAA, 0x55, 0xCC, 0x33])
        data += struct.pack('<HHI', self.__length, self.__type, self.__detections.getCRC32())
        data += self.__detections.to_Serial()
        return data

class V5SerialComms:

    __MAP_PACKET_TYPE = 0x0001

    def __init__(self, port = None):
        # Initialize properties of V5SerialComms class, including port, started status, and lock
        self.__dev = port
        self.__started = False
        self.__ser = None
        self.__detections = AIRecord([])
        self.__detectionLock = Lock()

    def start(self):
        # Start serial communication thread
        self.__started = True
        self.__thread = threading.Thread(target=self.__run, args=())
        self.__thread.daemon = True
        self.__thread.start()

    def __run(self):
        while self.__started:  # Continue running while the thread is started
            port = self.__dev
            try:
                if(port == None):  # If the port is not specified
                    from serial.tools.list_ports import comports
                    # Find devices that match the V5 description
                    devices = [dev for dev in comports() if "V5" in dev.description and "User" in dev.description]
                    # self.devices = [dev for dev in comports()]
                    # print(self.devices)
                    if(len(devices) == 0):
                        print("No V5 Brain detected.")
                        time.sleep(1)
                        continue
                    else:
                        port = devices[0].device 
                    
                print("Connecting to ", port)

                # Establish serial connection with the port
                self.__ser = serial.Serial(port, 230400, timeout=10)
                self.__ser.flushInput()
                self.__ser.flushOutput()

                while self.__started:  # Continue reading while thread is started
                    # Read data from the serial port
                    data = self.__ser.readline().decode("utf-8").rstrip()
                    # print(data)
                    if(data == "AA55CC3301"):
                        #send data
                        self.__detectionLock.acquire()
                        myPacket = V5SerialPacket(self.__MAP_PACKET_TYPE, self.__detections)
                        self.__detectionLock.release()
                        data = myPacket.to_Serial()
                        self.__ser.write(data)  # Write serialized data to the serial port


            # To close the serial port gracefully, use Ctrl+C to break the loop
            except serial.SerialException as e:
                print("Could not connect to ", port, ". Exception: ", e)
                time.sleep(1)    # Wait for 1 second before retrying
        
            if(self.__ser.isOpen()):
                self.__ser.close()    # Close the serial port if open

        print("V5SerialComms thread stopped.")

    def setDetectionData(self, data: AIRecord):
        # Aquire lock and set detection data
        self.__detectionLock.acquire()
        self.__detections = data
        self.__detectionLock.release()

    def stop(self):
        # Stop the thread by setting started flag to False and join the thread
        self.__started = False
        self.__thread.join()

    def __del__(self):
        # Destructor to call the stop method when the object is deleted
        self.stop