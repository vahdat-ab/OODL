This source code is related to the paper titled ["Detecting Out-of-Distribution Inputs in Deep Neural Networks Using an Early-Layer Output"](https://arxiv.org/abs/1910.10307)

## Software dependencies
The source code is written in Python with Tensorflow (including Keras) and Jupyter. Please make sure you have these tools installed properly in your system. 
- The version of Python used: 3.6.9
- The version of Tensorflow used: 2.1.0
- The version of the notebook server is: 6.0.1

**NOTE**: once you run *"download_required_files.ipynb"*, it will install require python packages.

## Datasets and models
The models and OOD datasets can be downloaded automatically by opening *"download_required_files.ipynb"* and executing all cells. If this process goes well, there will be two folders named *"ood_datasets"* and *"saved_models"* beside the file *"download_required_files.ipynb"*.

## Experiments
To experiment with our approaches and others listed in the paper, please open *"Performance_evaluation.ipynb"* and execute cells to get the results based on the metrics defined in the paper. If you have your own model and want to find the OODL and repeat the experiments, you first need to use *"find_oodl.ipynb"* for find the OODL for your model.

- By  *id_name*, you can set the ID dataset. The list of ID datasets is: ("MNIST", "CIFAR10", "CIFAR100")
- By  *id_model*, you can set the ID model. The list of ID models is: ("CUSTOM-MNIST", "VGG16", "ResNet-V1-44")
- By  *ood_appr_name*, you can select the OOD detection approach. The list of OOD detection approaches is: ("ours", "ours_w_p" , "softmax", "odin_wo_p", "odin_w_p", "mah_dist_logits", "mah_dist_logits_w_p", "entropy", "margin", "mc_dropout","mutual_info")

