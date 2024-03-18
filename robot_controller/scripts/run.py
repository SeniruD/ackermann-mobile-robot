#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config

import rospy 

def main():

    rospy.init_node('vlnce')
    rospy.loginfo("vlnce node started")

    run_type = "single_inference"
    exp_config = "/home/senirud/catkin_ws1/src/robot_controller/scripts/vlnce_baselines/config/r2r_baselines/test_set_inference.yaml"
    opts = None
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME) 
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "single_inference":
        trainer.single_inference()


if __name__ == "__main__":
    main()

