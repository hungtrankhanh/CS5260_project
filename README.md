# TransGAN: Two Transformers Can Make One Strong GAN
Code used for [TransGAN: Two Transformers Can Make One Strong GAN](https://arxiv.org/abs/2102.07074). 

## News
[checkpoint](https://github.com/VITA-Group/TransGAN/blob/master/exps/celeba64_test.sh) for generating images on celeba64 dataset is released!

## Main Pipeline
![Main Pipeline](assets/TransGAN.png)

## Visual Results
![Visual Results](assets/Visual_results.png)

### prepare fid statistic file
 ```bash
mkdir fid_stat
 ```
Download the pre-calculated statistics
([Google Drive](https://drive.google.com/drive/folders/1UUQVT2Zj-kW1c2FJOFIdGdlDHA3gFJJd?usp=sharing)) to `./fid_stat`.

### Environment
```bash
pip install -r requirements.txt
```
Notice: Pytorch version has to be <=1.3.0 !

### Training
Coming soon

### Testing
Firstly download the checkpoint from ([Google Drive](https://drive.google.com/drive/folders/1Rv7ycxFKBzXPpoqw6bdjj0cNtmaei0lM?usp=sharing)) to `./pretrained_weight`
```bash
# cifar-10
sh exps/cifar10_test.sh

# stl-10
sh exps/stl10_test.sh

# celeba64
sh exps/celeba64_test.sh
```

## Acknowledgement
Codebase from [AutoGAN](https://github.com/VITA-Group/AutoGAN), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## Citation
if you find this repo is helpful, please cite
```
@article{jiang2021transgan,
  title={TransGAN: Two Transformers Can Make One Strong GAN},
  author={Jiang, Yifan and Chang, Shiyu and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2102.07074},
  year={2021}
}
```
