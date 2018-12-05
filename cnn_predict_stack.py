import data
import numpy as np
from scipy.io import savemat, loadmat
import datetime
import os
import pdb

# All the files to be prcocessed
inFiles = ('DukeTurtle_test.mat',)

# CNN classification model 
cnnModelDef = 'DukeTurtle_info.json'
cnnModelWeights = 'DukeTurtle_info.h5'
clab = ['0', 'Certain Turtle']

def process_file(inFile, outFile, model):

    # Check if there's chunks
    nChunks, clab = data.load_mat_chunk(inFile,chunkNum=None)
    if nChunks is None:
        print(datetime.datetime.now())
        print('Reading from '+inFile)
        stacks, labels, clab = data.load_mat_batch(inFile)
        stacks = np.float32(stacks) / 255
        print(datetime.datetime.now())
        print('{} samples read'.format(stacks.shape[0]))

        prob = model.predict_proba(stacks, batch_size=128, verbose=1)
        print(datetime.datetime.now())
    else:
        prob = np.zeros((0,model.output_shape[1]))
        for chunkNum in range(1, nChunks+1):
            print('Reading chunk '+str(chunkNum)+'/'+str(nChunks)+' from '+inFile) 
            stacks, labels = data.load_mat_chunk(inFile, chunkNum=chunkNum)
            stacks = np.float32(stacks) / 255
            print(stacks.shape[0], 'samples read') 

            prob = np.append(prob, model.predict_proba(stacks, batch_size=128, verbose=1), axis=0)

    return prob

model = data.load_model(cnnModelDef, cnnModelWeights)

for inFile in inFiles:
    outFile = inFile.replace('.mat','_cnnClass.mat')
    p = process_file(inFile, outFile, model)
    print('Saving to '+outFile)
    savemat(outFile, {'p': p, 'clab': clab})


