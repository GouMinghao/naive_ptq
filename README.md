# naive_ptq
This project aims to provide the **Most Simple Introduction** and **Most Basic Ideas** of how a neural network is quantized and deployed.

By following the guidance, you can know the complete pipeline for NN deployment.
It should be one of the most simple implementations although there might be some bugs.

In industrial applications, NN is usually trained using PyTorch with float point number.
But it is deployed on ASICs or programmable hardwares such as DSP and FPGA, on which only fixed point numbers are supported.
So the network needs to be quantized with some skills by which little accuracy is lost but huge benefit on cost is obtained.

In this project, a very simple quantization method on a very simple Convolution layer is used as an example.
It has three steps:
1. We obtain a simple python model for the network.
2. Simple post training quantization(PTQ) of the network.
3. Deploy the network with naive C++.

In real-world applications, the first step is to train network with python on float point.
The second and third steps are usually integrated in an SDK provided by the hardware manufacturer such as Qualcomm/Apple/NVIDIA.

## Pipelines
There are 3 steps as introduced above.

### Training part
Training part is skipped as most people are very familiar with this part.
In this project, random weights and random inputs are used instead.

### Quantization part
To understand the details of convolution, a simple numpy convolution is implemented.
Then the network is symmetrically quantized and parameter are dumped for further deployment.
To verify the consistency of the deployed model and trained model, ground truth input and output are also dumped.
The difference is printed to show the accuracy loss during quantization.

### Deploy part
A simple C++ program is implemented. It loads dumped parameter and does inference.
The consistency is also check by comparing the C++ result with numpy convolution result and torch convolution result.

## Examples
1. Run [`example.py`](py/example.py) to quantize the float point model and dump the result
```bash
cd py
python example.py
# you can also run some unit tests by
python run_unit_tests.py
```

2. Build C++ program and check consistency.
```bash
# install Eigen3 and cmake first
# apt install libeigen3-dev cmake
cd cpp
bash run.sh
```