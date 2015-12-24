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

import argparse

# model_name = re.sub(r"\.py$", "", os.path.basename(args.network))
# model_module = load_module(os.path.dirname(args.network), model_name)

# model = model_module.Network()
# serializers.load_hdf5(args.trained_model, model)

# def plotD(dim,data):
#     size = int(math.ceil(math.sqrt(dim[0])))
#     if(len(dim)==4):
#         for i,channel in enumerate(data):
#             ax = plt.subplot(size,size, i+1)
#             ax.xaxis.set_major_locator(NullLocator())
#             ax.yaxis.set_major_locator(NullLocator())
#             accum = channel[0]
#             for ch in channel:
#                 accum += ch
#             accum /= len(channel)
#             ax.imshow(accum, interpolation='nearest')
#     else:
#         plt.imshow(W.data, interpolation='nearest')

# def plot(W):
#     dim = eval('('+W.label+')')[0]
#     size = int(math.ceil(math.sqrt(dim[0])))
#     if(len(dim)==4):
#         for i,channel in enumerate(W.data):
#             ax = plt.subplot(size,size, i+1)
#             ax.xaxis.set_major_locator(NullLocator())
#             ax.yaxis.set_major_locator(NullLocator())
#             accum = channel[0]
#             for ch in channel:
#                 accum += ch
#             accum /= len(channel)
#             ax.imshow(accum, interpolation='nearest')
#     else:
#         plt.imshow(W.data, interpolation='nearest')

# def showPlot(layer):
#     plt.clf()
#     W = layer.params().next()
#     fig = plt.figure()
#     fig.patch.set_facecolor('black')
#     fig.suptitle(W.label, fontweight="bold",color="white")
#     plot(W)
#     plt.show()

# def showW(W):
#     plt.clf()
#     fig = plt.figure()
#     fig.patch.set_facecolor('black')
#     fig.suptitle(W.label, fontweight="bold",color="white")
#     plot(W)
#     plt.show()

# def getW(layer):
#     return layer.params().next()

# def savePlot2(layer):
#     plt.clf()
#     W = layer.params().next()
#     fig = plt.figure()
#     fig.patch.set_facecolor('black')
#     fig.suptitle(W.label, fontweight="bold",color="white")
#     plot(W)
#     plt.draw()
#     plt.savefig(W.label+".png")

# def savePlot(W,name):
#     plt.clf()
#     fig = plt.figure()
#     fig.suptitle(name+" "+W.label, fontweight="bold")
#     plot(W)
#     plt.draw()
#     plt.savefig(name+".png")

# def layers(model):
#     for layer in model.namedparams():
#         if layer[0].find("W") > -1:
#             print layer[0],layer[1].label
#             savePlot(layer[1],layer[0].replace("/","_"))

# def layersName(model):
#     for layer in model.namedparams():
#         print layer[0],layer[1].label

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
                ax.imshow(accum, interpolation='nearest') # interpolationはセットしないほうがいい？
        else:
            plt.imshow(W.data, interpolation='nearest')
        
    def save_plot(self, W, name):
        plt.clf()
        fig = plt.figure()
        fig.suptitle(name + " " + W.label, fontweight='bold')
        self.plot(W)
        plt.draw()
        plt.savefig(self.output_dir + os.sep + name + ".png")
        
    def visualize_first_conv_layer(self):
        for layer in sorted(self.model.namedparams()):
            if (layer[0].find('W') > -1) and (layer[0].find('conv') > -1):
                print layer[0], layer[1].label
                self.save_plot(layer[1], layer[0].replace('/', '_'))
                break
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize layer')
    parser.add_argument('network', help='Path to Network')
    parser.add_argument('trained_model', help='Path to trained model')
    parser.add_argument('output_dir', help="Path to output")
    args = parser.parse_args() 

    v = LayerVisualizer(args.network, args.trained_model, args.output_dir)
    v.layers()

