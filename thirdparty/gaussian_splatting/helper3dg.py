#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
from random import randint
import random
import sys
import uuid
import time
import json

import numpy as np
import cv2
from tqdm import tqdm
import shutil

sys.path.append("./thirdparty/gaussian_splatting")

from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def getparser():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) #we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-2)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10000, 12000, 25_000, 30_000])
    parser.add_argument("--test_iterations", default=-1, type=int)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=5, help="5 debug , 50 used")
    parser.add_argument("--basicfunction", type=str, default = "gaussian")
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v2")
    parser.add_argument("--configpath", type=str, default = "None")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    return args, lp.extract(args), op.extract(args), pp.extract(args)

def getrenderparts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]




def gettestparse():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v3")
    parser.add_argument("--valloader", type=str, default = "colmap")
    parser.add_argument("--configpath", type=str, default = "1")

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # configpath
    safe_state(args.quiet)

    multiview = True if args.valloader.endswith("mv") else False

    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
        print("args: " + str(args))

    return args, model.extract(args), pipeline.extract(args), multiview
