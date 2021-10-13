# Single-Stage Semantic Segmentation from Image Labels

Forked from https://github.com/visinf/1-stage-wseg

### Setup
0. **Minimum requirements.** This project was originally developed with Python 3.6, PyTorch 1.0 and CUDA 9.0. The training requires at least two Titan X GPUs (12Gb memory each).
1. **Setup your Python environment.** Please, clone the repository and install the dependencies. We recommend using Anaconda 3 distribution:
    ```
    conda create -n <environment_name> --file requirements.txt
    ```
2. **Download and link to the dataset.** We train our model on the original Pascal VOC 2012 augmented with the SBD data (10K images in total). Download the data from:
    - VOC: [Training/Validation (2GB .tar file)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    - SBD: [Training (1.4GB .tgz file)](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

    Link to the data:
    ```
    ln -s <your_path_to_voc> <project>/data/voc
    ln -s <your_path_to_sbd> <project>/data/sbd
    ```
    Make sure that the first directory in `data/voc` is `VOCdevkit`; the first directory in `data/sbd` is `benchmark_RELEASE`.
3. **Download pre-trained models.** Download the initial weights (pre-trained on ImageNet) for the backbones you are planning to use and place them into `<project>/models/weights/`.

    | Backbone | Initial Weights | Comment |
    |:---:|:---:|:---:|
    | WideResNet38 | [ilsvrc-cls_rna-a1_cls1000_ep-0001.pth (402M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth) | Converted from [mxnet](https://github.com/itijyou/ademxapp) |
    | VGG16 | [vgg16_20M.pth (79M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/vgg16_20M.pth) | Converted from [Caffe](http://liangchiehchen.com/projects/Init%20Models.html) |
    | ResNet50 | [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) | PyTorch official |
    | ResNet101 | [resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | PyTorch official |


### Training, Inference and Evaluation
The directory `launch` contains template bash scripts for training, inference and evaluation. 

**Training.** For each run, you need to specify names of two variables, for example
```bash
EXP=baselines
RUN_ID=v01
```
Running `bash ./launch/run_voc_resnet38.sh` will create a directory `./logs/pascal_voc/baselines/v01` with tensorboard events and will save snapshots into `./snapshots/pascal_voc/baselines/v01`.

**Inference.** To generate final masks, please, use the script `./launch/infer_val.sh`. You will need to specify:
* `EXP` and `RUN_ID` you used for training;
* `OUTPUT_DIR` the path where to save the masks;
* `FILELIST` specifies the file to the data split;
* `SNAPSHOT` specifies the model suffix in the format `e000Xs0.000`. For example, `e020Xs0.928`;
* (optionally) `EXTRA_ARGS` specify additional arguments to the inference script.

**Evaluation.** To compute IoU of the masks, please, run `./launch/eval_seg.sh`. You will need to specify `SAVE_DIR` that contains the masks and `FILELIST` specifying the split for evaluation.

### Pre-trained model
For testing, we provide our pre-trained WideResNet38 model:

| Backbone | Val | Val (+ CRF) | Link |
|:---:|:---:|:---:|---:|
| WideResNet38 | 59.7 | 62.7 | [model_enc_e020Xs0.928.pth (527M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/model_enc_e020Xs0.928.pth) |

The also release the masks predicted by this model:

| Split | IoU | IoU (+ CRF) | Link | Comment |
|:---:|:---:|:---:|:---:|:---:|
| train-clean (VOC+SBD) | 64.7 | 66.9 | [train_results_clean.tgz (2.9G)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/results/train_results_clean.tgz) | Reported IoU  is for VOC |
| val-clean | 63.4 | 65.3 | [val_results_clean.tgz (423M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/results/val_results_clean.tgz)  | |
| val | 59.7 | 62.7 | [val_results.tgz (427M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/results/val_results.tgz) | |
| test | 62.7 | 64.3 | [test_results.tgz (368M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/results/test_results.tgz) | |

The suffix `-clean` means we used ground-truth image-level labels to remove masks of the categories not present in the image.
These masks are commonly used as pseudo ground truth to train another segmentation model in fully supervised regime.

```
@inproceedings{Araslanov:2020:WSEG,
  title     = {Single-Stage Semantic Segmentation from Image Labels},
  author    = {Araslanov, Nikita and and Roth, Stefan},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```
