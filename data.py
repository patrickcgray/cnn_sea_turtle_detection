from __future__ import absolute_import
from keras.models import model_from_json
import numpy as np
import os
from scipy.io import loadmat
import hdf5storage as h5
import tables

def load_model(defFile, weightFile):
    # as per http://keras.io/faq/#how-can-i-save-a-keras-model

    model = model_from_json(open(defFile).read())
    model.load_weights(weightFile)

    return model

def cellstr_from_tables(f):

    data = []
    for entity in f:
        data.append(''.join([chr(x[0]) for x in entity[0]]))

    return data

def load_mat_batch(filePath, source='mat'):

    # Labels may or may not exist
    labels = None
    labelNames = None

    with tables.open_file(filePath) as f:
        if 'nChunk' in f.root._v_leaves.keys():
            nChunk = f.root.nChunk.read()
            data = f.root.features_1.read()
            for ch in range(2, nChunk+1):
                feaName = 'features_{}'.format(ch)
                data = np.append(data, f.root._v_leaves[feaName].read(), axis=0)

            if 'labels_1' in f.root._v_leaves.keys():
                labels = f.root.labels_1.read().flatten()
                for ch in range(2, nChunk+1):
                    labName = 'labels_{}'.format(ch)
                    labels = np.append(labels, f.root._v_leaves[labName].read().flatten(), axis=0)

        else:
            data = f.root.features.read()
            if 'labels' in f.root._v_leaves.keys():
                labels = f.root.labels.read().flatten()

        if source == 'mat':
            data = np.transpose(data)
            data = np.rollaxis(data, 3)

        if 'labelNames' in f.root._v_leaves.keys():
            labelNames = cellstr_from_tables(f.root.labelNames.read())

    return data, labels, labelNames

def load_mat_chunk(filePath,chunkNum=None):
    if chunkNum is None:
        try:
            nChunk = h5.loadmat(filePath, variable_names=['nChunk'])['nChunk']
        except h5.lowlevel.CantReadError:
            nChunk = None

        try:
            struct = h5.loadmat(filePath, variable_names=['labelNames'])
            labelNames = [f[0][0] for f in struct['labelNames'][0]]
        except:
            labelNames = None

        return nChunk, labelNames

    var = 'features_'+str(chunkNum)
    data = h5.loadmat(filePath, variable_names=[var])[var]
    data = np.rollaxis(data, 3)

    try:
        var = 'labels_'+str(chunkNum)
        labels = h5.loadmat(filePath, variable_names=[var])[var]
    except:
        labels = None

    return data, labels


