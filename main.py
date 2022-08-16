#!/usr/bin/env python
from scbasset.model_class import ModelClass, Config
import configargparse
from torch import cuda
from icecream import ic


def main():
    config = Config()
    config.make_parser()
    config.load_args()

    cuda.set_device(config.cuda)
    ic(cuda.current_device())

    dashboard = ModelClass(config=config)
    transfer_learning = False if config.weights is None else True
    dashboard.activate_training(transfer_learning)
    dashboard.get_model_summary()
    # if transfer_learning:
    #     ic(transfer_learning)
    #     dashboard.feature_extract(device='gpu')
    dashboard.fit()

if __name__ == "__main__":
    main()
