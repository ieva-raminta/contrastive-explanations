import torch
import torch.nn as nn
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
import numpy as np

model_path = "/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/results/Outcome/joint_model/legal_bert/facts/ccc660d6049c4d1782bc6c81f2f30b12/model.pt"
model = torch.load(model_path)

model.eval()

torch.manual_seed(123)
np.random.seed(123)

input = (torch.rand(2, 3), torch.rand(2, 3), torch.rand(2, 3))
baseline = (torch.zeros(2, 3), torch.rand(2, 3), torch.rand(2, 3))

ig = IntegratedGradients(model)
attr, delta = ig.attribute(input, baseline, return_convergence_delta=True)
print(attr)
print(delta)
