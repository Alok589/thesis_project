import numpy as np
import pandas as pd

import torch

model = nn.Sequential(
    nn.conv2d(64, 3, 3)
    nn.Relu(inplace = True)
    nn.Drouout(0.5)
)

