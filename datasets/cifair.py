""" ciFAIR data loaders for PyTorch.

Version: 1.0

https://cvjena.github.io/cifair/
"""

import torchvision.datasets

class ciFAIR10(torchvision.datasets.CIFAR10):
    base_folder = 'ciFAIR-10'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    filename = 'ciFAIR-10.zip'
    tgz_md5 = 'ca08fd390f0839693d3fc45c4e49585f'
    test_list = [
        ['test_batch', '01290e6b622a1977a000eff13650aca2'],
    ]

class ciFAIR100(torchvision.datasets.CIFAR100):
    base_folder = 'ciFAIR-100'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip'
    filename = 'ciFAIR-100.zip'
    tgz_md5 = 'ddc236ab4b12eeb8b20b952614861a33'
    test_list = [
        ['test', '8130dae8d6fc6a436437f0ebdb801df1'],
    ]