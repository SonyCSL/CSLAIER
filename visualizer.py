# -*- coding: utf-8 -*-

import os
import imp
import re

import chainer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
from chainer.links import caffe
from matplotlib.ticker import * 
from chainer import serializers

float32 = 0

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol, [dir_name])
    return imp.load_module(symbol, file, path, description)

class LayerVisualizer:
    def __init__(self, network_path, trained_model_path, output_dir):
        model_name = re.sub(r"\.py$", "", os.path.basename(network_path))
        model_module = load_module(os.path.dirname(network_path), model_name)
        self.model = model_module.Network()
        serializers.load_hdf5(trained_model_path, self.model)
        self.output_dir = output_dir
        
    def plot(self, W):
        dim = eval('(' + W.label + ')')[0]
        size = int( math.ceil(math.sqrt(dim[0])))
        if len(dim) == 4:
            for i, channel in enumerate(W.data):
                ax = plt.subplot(size, size, i+1)
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
                accum = channel[0]
                for ch in channel:
                    accum += ch
                accum /= len(channel)
                ax.imshow(accum)
        else:
            plt.imshow(W.data)
        
    def save_plot(self, W, name):
        plt.clf()
        fig = plt.figure()
        fig.suptitle(name + " " + W.label, fontweight='bold', color='#ffffff')
        self.plot(W)
        plt.draw()
        plt.savefig(self.output_dir + os.sep + name + ".png", facecolor="#001100")
        
    def visualize_first_conv_layer(self, name=None):
        candidate = None
        for layer in sorted(self.model.namedparams()):
            if layer[0].find('W') > -1:
                if layer[0].find('conv') > -1:
                    candidate = layer
                    break
                if candidate is None:
                    candidate = layer
                   
        if candidate is not None:
            if name is None:
                name = candidate[0].replace('/', '_')
            self.save_plot(candidate[1], name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='visualize layer')
    parser.add_argument('network', help='Path to Network')
    parser.add_argument('trained_model', help='Path to trained model')
    parser.add_argument('output_dir', help="Path to output")
    args = parser.parse_args() 

    v = LayerVisualizer(args.network, args.trained_model, args.output_dir)
    v.visualize_first_conv_layer()

