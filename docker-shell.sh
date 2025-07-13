#!/bin/bash
sg docker -c "docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace hiyouga/verl:ngc-th2.7.0-cu12.6-vllm0.9.1 bash"