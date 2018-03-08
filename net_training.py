# Copyright (c) 2018 Alaa BEN JABALLAH

from __future__ import print_function
import sys
import os
import time
import cv2
import numpy as np
import theano
import theano.tensor as T
import lasagne

size = 96
input_var = T.tensor4('inputs')

def load_img(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE #cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        small = cv2.resize(gray[y:y+h, x:x+w], (size,size))
        arr = np.array(small / np.float(256))
        arr = arr.reshape(1,1,size,size)
        return arr
    return np.array([])

# modul przygotowywania danych
def train_net(datadir, imagedir, labeldir, network, epochs=1000, save_each=False):
    import gzip
    def load(img_path): 
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            small = cv2.resize(gray[y:y+h, x:x+w], (size,size))
            return small / np.float(256)

    def load_images(path): 
        paths = os.listdir(path)
        files = []
        for p in paths :
            if os.path.isfile(path + "\\" + p) and (p.endswith(".jpg") or p.endswith(".png")):
                files.append(p)
        data = []
        for f in files :
            face = load(path + "\\" + f) 
            if face is not None :
                data.append(face)
        return data

    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if len(inputs) < batchsize :
            batchsize = len(inputs)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            out1 = []
            out2 = []
            for i in excerpt :
                out1.append(inputs[i])
                out2.append(targets[i])
            out1_arr = np.array(out1)
            out1_arr = out1_arr.reshape(-1,1,size,size)
            yield out1_arr, np.array(out2)

    # ladowanie danych
    print("Loading data...")
    data = []
    labels = []
    imagepaths = os.listdir(datadir + imagedir)
    labelpaths = os.listdir(datadir + labeldir)
    counter = 0
    for path in labelpaths :
        count = 100 * counter / len(labelpaths)
        out = '#' * (count / 5) + ' ' * (20 - (count / 5)) + ' ' + str(count) + '%'
        sys.stdout.write('\r'+out)
        if os.path.isdir(datadir + labeldir + path):
            in_dir = os.listdir(datadir + labeldir + path)
            for in_path in in_dir :
                if os.path.isdir(datadir + labeldir + path + '\\' + in_path):
                    txt_dir = os.listdir(datadir + labeldir + path + '\\' + in_path)
                    for textfile in txt_dir:
                        lab = 0
                        if os.path.isfile(datadir + labeldir + path + '\\' + in_path + '\\' + textfile) and textfile.endswith(".txt"):
                            with open(datadir + labeldir + path + '\\' + in_path + '\\' + textfile) as f:
                                for line in f :
                                    lab = int(float(line))
                            imgs = load_images(datadir + imagedir + path + '\\' + in_path + '\\')
                            for i in range (0, len(imgs)):
                                labels.append(lab - 1)
                            data.extend(imgs)
                        break
        counter += 1

    data_arr = np.array(data)
    print (len(data_arr))
    data_arr = data_arr.reshape(-1,1,size,size)
    labels_arr = np.array(labels)
    print ('\ndata loaded')

    target_var = T.ivector('targets')

    # funkcja bledu
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # funkcja dobierajaca kolejne wagi Stochastic Gradient Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # funkcja bledu
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # petla trenujaca
    print("Starting training...")
    for epoch in range(epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(data, labels, 50, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        if save_each:
            name = "custom_model%s.npz" % epoch
            np.savez(name, *lasagne.layers.get_all_param_values(network))

    np.savez('custom_model.npz', *lasagne.layers.get_all_param_values(network))

# modul tworzenia architektury sieci
def build_cnn(model = None):
    network = lasagne.layers.InputLayer(shape=(None, 1, size, size),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=7,
            nonlinearity=lasagne.nonlinearities.softmax)

    # ladowanie wag sieci jezeli podano
    if model != None:
        with np.load(model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    return network

# ewaluacja sieci
# 0=anger, 1=contempt, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise
def evaluate(network, faces_matrix):
    if faces_matrix.ndim != 4 :
        return np.array([[.0,.0,.0,.0,.0,.0,.0]])

    out = lasagne.layers.get_output(network, faces_matrix)
    return out.eval()
