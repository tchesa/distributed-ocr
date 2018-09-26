import os, socket, sys, Pyro4, Pyro4.socketutil

from glob import glob
import cv2
import re
import numpy as np
from sklearn import svm
from skimage.feature import hog
import pickle, time, codecs

DIR = "/home/cesar/Desktop/database/" # linux
default_resolution = (24, 32)

letters = "abcdefghijklmnopqrstuvwxyz"
digits = "0123456789"

@Pyro4.behavior(instance_mode="single")
class Forecaster(object):

    letters = ""
    digits = ""

    svms = {}

    @Pyro4.expose
    def setChars(self, chars):
        for c in chars:
            if c in letters:
                self.letters += c
            else:
                self.digits += c
        print('letters:', self.letters)
        print('digits:', self.digits)
        self.init()
        # return 1

    def init(self):
        for c in self.letters + self.digits:
            self.svms[c] = svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.5)
        print('initialized')

    @Pyro4.expose
    def train(self):
        tracks = glob(DIR + "training/*/")
        
        samples = {}
        responses = {}
        positives = {}

        for c in self.letters + self.digits:
            samples[c] = []
            responses[c] = []
            positives[c] = 0

        print("get data for training...")
        # the training/test database is divided by tracks
        # each track have a collection of frames (images)
        for track in tracks:
            files = glob(track + "/*.png")
            for f in files: # each image of the track
                img = cv2.imread(f, 0) # read image
                notes = parseNotations(f.replace(".png", ".txt")) # get image notes
                text = notes["text"].replace("-","").lower() # characters of the plate
                for i in range(len(text)):
                    col = self.letters if (i < 3) else self.digits
                    rect = notes["position_chars"][i]
                    for c in col:
                        response = 1 if (c == text[i]) else -1 # one against all
                        if (c == text[i]):
                            positives[c] += 1
                        nimg = cv2.resize(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], default_resolution) # normalize images to hog describer
                        hist = hog(nimg, block_norm='L2-Hys') # hog describer
                        samples[c].append(hist)
                        responses[c].append(response)
                        
        print("start training...")
        for c in self.letters + self.digits:
            trainData = np.float32(samples[c]) # Convert objects to Numpy Objects
            labels = np.array(responses[c])
            self.svms[c].fit(trainData, labels) # train svm
        return 'training finished'

    @Pyro4.expose
    def forecast(self, serial, isLetter):
        hist = pickle.loads(codecs.decode(serial.encode(), "base64"))
        testData = np.float32([hist])
        results = []
        col = self.letters if isLetter else self.digits
        for c in col:
            results.append((c, self.svms[c].predict_proba(testData)[0][1]))
        return results
        # hist = np.float32(pickle.loads(data))
        # print("hist: ", hist)

def parseNotations(location): # parse notes file
    chars = []
    text_pattern = re.compile("text")
    plate_pattern = re.compile("position_plate")
    chars_pattern = re.compile("char[0-9]")
    with open(location, "rb") as f:
        for line in f:
            line = line.decode("utf-8").strip()
            if text_pattern.match(line):
                text = line.replace("text:","").strip()
            elif plate_pattern.match(line):
                numbers = line.replace("position_plate:", "").strip().split(" ")
                plate_position = tuple(map(lambda x: int(x), numbers))
            elif chars_pattern.match(line):
                numbers = line[7:].strip().split(" ")
                chars.append(tuple(map(lambda x: int(x), numbers)))
        f.close()
        return {"text": text, "position_plate": plate_position, "position_chars": chars}

with Pyro4.Daemon(host=Pyro4.socketutil.getIpAddress(None)) as daemon:
    # create a unique name for this worker (otherwise it overwrites other workers in the name server)
    worker_name = "Worker_%d@%s" % (os.getpid(), socket.gethostname())
    print("Starting up worker", worker_name)
    uri = daemon.register(Forecaster)
    with Pyro4.locateNS() as ns:
        ns.register(worker_name, uri, metadata={"worker.forecaster"})
    daemon.requestLoop()
