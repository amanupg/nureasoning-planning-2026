#!/bin/bash
# Unpack the Edge env on a pod (paths mirror Insomnia) and smoke-test imports + GPU.
N=/path/to/cluster/nureasoning
cd $N/ext && [ -d edge_env/bin ] || tar xf edge_env.tar --no-same-owner
cd /path/to/cluster/jepa-nureasoning
$N/ext/edge_env/bin/python -I -c "
import torch, transformers, alpamayo2_edge, sys
print('py', sys.version.split()[0], 'torch', torch.__version__, 'tf', transformers.__version__)
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); y = x @ x; torch.cuda.synchronize()
print('cuda ok', torch.cuda.get_device_name(), 'cap', torch.cuda.get_device_capability())
" && echo SETUP_OK || echo SETUP_FAIL
