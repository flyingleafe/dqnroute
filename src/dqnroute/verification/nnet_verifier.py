import subprocess
import re
import os

from typing import *

import numpy as np
import torch

from .ml_util import Util

os.chdir("../NNet")
from utils.writeNNet import writeNNet
os.chdir("../src")

class NNetVerifier:
    def __init__(self, marabou_path: str, network_filename: str, property_filename: str):
        self.marabou_path = marabou_path
        self.network_filename = network_filename
        self.property_filename = property_filename
    
    def verify_adv_robustness(self, net: torch.nn.Module, weights: List[torch.tensor],
                              biases: List[torch.tensor], input_center: torch.tensor, input_eps: float,
                              output_constraints: List[str], check_or: bool) -> bool:
        if check_or:
            # run separately for each constraint
            falsified = False
            for constraint in output_constraints:
                result = self.verify_adv_robustness(
                    net, weights, biases, input_center, input_eps, [constraint], False
                )
                falsified = falsified or not result
                if falsified:
                    break
            if not falsified:
                print("  no counterexample")
            return
        
        output_dim = biases[-1].shape[0]
        
        # write the NN
        input_dim = weights[0].shape[1]
        input_mins  = list(np.zeros(input_dim) - 10e6)
        input_maxes = list(np.zeros(input_dim) + 10e6)
        means = list(np.zeros(input_dim)) + [0.]
        ranges = list(np.zeros(input_dim)) + [1.]
        writeNNet([Util.to_numpy(x) for x in weights],
                  [Util.to_numpy(x) for x in biases],
                  input_mins, input_maxes, means, ranges, self.network_filename)
        # write the property
        with open(self.property_filename, "w") as f:
            for i in range(input_dim):
                f.write(f"x{i} >= {input_center[i] - input_eps}{os.linesep}")
                f.write(f"x{i} <= {input_center[i] + input_eps}{os.linesep}")
            for constraint in output_constraints:
                f.write(constraint + os.linesep)
    
        # call Marabou
        command = [self.marabou_path, self.network_filename, self.property_filename]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = []
        while process.poll() is None:
            line = process.stdout.readline().decode("utf-8")
            #print(line, end="")
            lines += [line.strip()]
        final_lines = process.stdout.read().decode("utf-8").splitlines()
        process.stdout.close()
        for line in final_lines:
            #print(line)
            lines += [line.strip()]

        # construct counterexample
        xs = np.zeros(input_dim)  + np.nan
        ys = np.zeros(output_dim) + np.nan
        for line in lines:
            tokens = re.split(" *= *", line)
            if len(tokens) == 2:
                #print(tokens)
                value = float(tokens[1])
                index = int(tokens[0][1:]) 
                if tokens[0].strip().startswith("x"):
                    xs[index] = value
                elif tokens[0].strip().startswith("y"):
                    ys[index] = value
                else:
                    raise RuntimeError(f"Unexpected assignment {line}.")
        assert     np.isnan(xs).all() and     np.isnan(ys).all() or \
               not np.isnan(xs).any() and not np.isnan(ys).any(), \
               f"Problems with reading a counterexample (xs={xs}, ys={ys}, lines={lines})!"
        if not np.isnan(ys).any():
            with torch.no_grad():
                actual_output = Util.to_numpy(net(Util.conditional_to_cuda(torch.tensor(xs))))
            assert (np.abs(ys - actual_output) < 0.01).all(), "Output cross-check failed!"
            print(f"  counterexample: NN({list(xs)}) = {ys} [cross-check: {actual_output}]")
            return False
        else:
            return True