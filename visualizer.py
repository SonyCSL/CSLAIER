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
        output_file_name = name.replace('/', '_') + '.png'
        plt.savefig(self.output_dir + os.sep + output_file_name, facecolor="#001100")
        return output_file_name

    def get_layer_list(self):
        layers = []
        for layer in sorted(self.model.namedparams()):
            if layer[0].find("W") > -1:
                layers.append({"name": layer[0], "params": layer[1].label})
        return layers
        
    def visualize_all(self):
        for layer in sorted(self.model.namedparams()):
            if layer[0].find("W") > -1:
                self.save_plot(layer[1], layer[0])

    def visualize(self, layer_name):
        output_file_name = None
        for layer in sorted(self.model.namedparams()):
            if layer[0].find(layer_name) > -1:
                output_file_name = self.save_plot(layer[1], layer[0])
                break
        return output_file_name
                
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='visualize layer')
    parser.add_argument('network', help='Path to Network')
    parser.add_argument('trained_model', help='Path to trained model')
    parser.add_argument('output_dir', help="Path to output")
    args = parser.parse_args() 

    v = LayerVisualizer(args.network, args.trained_model, args.output_dir)

