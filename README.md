# Multi-layer-preceptron from scratch
This is a simple implementation of a multi-layer perceptron from scratch using numpy. The implementation is tested on the MNIST dataset.

## Quick start
1. Clone this repository
```bash
git clone https://github.com/shaohsuanhung/neural-network-from-scratch.git
```
2. Prepare the environment, since the code implement the MLP only using the basic library. All you need is python and numpy!
```bash
conda create --name nn_scratch python=3.8
conda activate nn_scratch
```

3. Prepare the MNIST dataset in CSV format, you can download the datast from [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

4. Run the script to train the model (10 epochs by default)
```bash
python mlp.py
```