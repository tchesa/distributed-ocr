# python3 -m Pyro4.naming [-n name]

import Pyro4, random, Pyro4.errors
from time import sleep
import pickle, codecs, time, re

from glob import glob
import cv2
import numpy as np
from sklearn import svm
from skimage.feature import hog

DIR = "/home/cesar/Desktop/database/" # linux
letters = "abcdefghijklmnopqrstuvwxyz"
digits = "0123456789"
allchars = letters + digits

def splitChars(workers):
    start = time.time()
    groups = ["" for x in range(len(workers))]
    for i in range(len(allchars)):
        groups[i%len(workers)] += allchars[i]
    # results = []
    for i in range(len(workers)):
        # results.append(workers[i].setChars(groups[i]))
        workers[i].setChars(groups[i])
    print("phase0: {}".format(time.time() - start))
    # for r in results:
    #     print(r.value)
    #     sleep(1)

def train(workers):
    start = time.time()
    for w in workers:
        w._pyroAsync() # set proxy in asynchronous mode

    results = [worker.train() for worker in workers]
    # for w in workers:
    #     results.append(w.train())

    # while False in [r.ready for r in results]:
        # print([r.ready for r in results])
    for r in results:
        # r.wait()
        print(r.value) # waiting for results
    print("phase1: {}".format(time.time() - start))

def test(workers):
    start = time.time()
    for w in workers:
        w._pyroAsync(asynchronous=False) # set proxy in asynchronous mode

    tracks = glob(DIR + "training/*/")
    responses = {}

    n_tests = 0
    n_errors = 0
    confusion = [[0 for x in range(len(letters + digits))] for y in range(len(letters + digits))]
    conf_index = {}
    for i in range(len(letters + digits)):
        conf_index[(letters + digits)[i]] = i

    print("get data for testing...")
    for track in tracks:
        files = glob(track + "/*.png")
        for f in files: # each image of the track
            img = cv2.imread(f, 0) # read the image
            notes = parseNotations(f.replace(".png", ".txt")) # get image notes
            text = notes["text"].replace("-","").lower() # plate characters
            for i in range(0,len(text)):
                rect = notes["position_chars"][i]
                nimg = cv2.resize(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], (24, 32)) # normalize images
                hist = hog(nimg, block_norm='L2-Hys')
                serial = codecs.encode(pickle.dumps(hist), "base64").decode()
                results = []
                # for j in range(len(workers)):
                for worker in workers:
                    _results = worker.forecast(serial, (i < 3))
                    # print(_results)
                    results += _results

                predicted = oneAgainstAll(results) # get final answer using one-against-all
                confusion[conf_index[text[i]]][conf_index[predicted]] += 1 # feed confusion matrix
                n_tests += 1
                n_errors += 0 if (predicted == text[i]) else 1 # compare real vs. predicted
            
    print("{} tests. {} mistakes.".format(n_tests, n_errors))
    m = np.array(confusion, np.int32)
    print("phase2: {}".format(time.time() - start))

def oneAgainstAll(results):
    better = 0
    for i in range(0, len(results)):
        if results[i][1] > results[better][1]:
            better = i
    return results[better][0]

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

def main():
    with Pyro4.locateNS() as ns:
        all_counters = ns.list(prefix="Worker") # gets a dictionary of servers
        workers = [Pyro4.Proxy(uri) for uri in all_counters.values()]

        splitChars(workers)
        train(workers)
        test(workers)

main()
