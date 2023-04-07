# organick - *Healthy Plants Watering Can*
<sub>Columbia University 2022-2023 Mechanical Engineering Undergraduate Senior Design Team 9</sub>

The project centers around a semi-smart watering can that helps the user tend to their plants. Instead of fully automating this process, which is a well-researched problem, we provide an interface to detect and treat basic ailments using four common treatment liquids.

In terms of software, this includes a program to control an Nvidia Jetson Nano mounted inside a watering can. The Jetson runs a camera and a neural network to perform detection. It controls fluid drip valves through GPIO while also providing a user interface. A Jupyter Notebook is provided to facilitate model training based on the [iNaturalist dataset](https://github.com/visipedia/inat_comp/tree/master/2021). This allows for identification of plants, insects, arachnids, and fungi, among many species.

## Setup
Install a conda environment from one of the provided `.yml` files based on whether you're training the neural network or just running inference. The `dev` one takes quite a while. You might get errors installing the pip packages `nvidia-tensorrt` and `onnxruntime-gpu`, but these shouldn't stop installation. Add these later if needed.

Initialize the [camera](https://www.arducam.com/product/b0183-arducam-imx219-distortioin-m12-mount-camera-module-raspberry-pi-compute-module/) interface on the Jetson using [these](https://docs.arducam.com/Nvidia-Jetson-Camera/Native-Camera/Quick-Start-Guide/) instructions.
