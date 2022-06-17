
from util import helps
from resModel import train
import numpy as np
import joblib
from cec2013lsgo.cec2013 import Benchmark
from os import path
from Grouping import Proposal, Comparison
from DE import DE
import os


def write_obj(data, path):
    with open(path, 'a') as f:
        f.write(str(data) + ', ')
        f.write('\n')
        f.close()


if __name__ == '__main__':

    Dim = 2000
    sample_size = 3
    bias = 1
    dense_size = 200000
    this_path = path.dirname(path.realpath(__file__))

    bench = Benchmark()
    trial_run = 25
    FEs = 6000000
    NIND = 30
    for func_num in range(1, 12):
        func = bench.get_function(func_num)
        info = bench.get_info(func_num)
        scale_range = [info["lower"], info["upper"]]
        """Training model"""
        # for i in range(sample_size):
        #     train.Model_Build(Dim, func_num, bias, dense_size, this_path, sample_size)

        """Load model and apply LIEM"""
        # LIEM_groups = Proposal.LIEM_model(Dim, func_num, bias, dense_size, this_path, sample_size)
        for i in range(trial_run):
            G_groups = Comparison.DECC_G(Dim, groups_num=40, max_number=50)
            Max_iter = int(FEs / NIND / Dim) - 2
            var_trace, obj_trace = DE.CC(Dim, NIND, Max_iter, func, scale_range, G_groups)
            obj_path = this_path + '/data/obj/proposal/f' + str(func_num)
            write_obj(obj_trace, obj_path)

