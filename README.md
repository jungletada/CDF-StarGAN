
# StarGAN v2 for CFD
## - Official PyTorch Implementation


> **StarGAN v2: Diverse Image Synthesis for Multiple Domains**<br>
> [Yunjey Choi](https://github.com/yunjey)\*, [Youngjung Uh](https://github.com/youngjung)\*, [Jaejun Yoo](http://jaejunyoo.blogspot.com/search/label/kr)\*, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<br>
> In CVPR 2020. (* indicates equal contribution)<br>


> **Abstract:** *A good image-to-image translation model should learn a mapping between different visual domains while satisfying the following properties: 1) diversity of generated images and 2) scalability over multiple domains. Existing methods address either of the issues, having limited diversity or multiple models for all domains. We propose StarGAN v2, a single framework that tackles both and shows significantly improved results over the baselines. Experiments on CelebA-HQ and a new animal faces dataset (AFHQ) validate our superiority in terms of visual quality, diversity, and scalability. To better assess image-to-image translation models, we release AFHQ, high-quality animal faces with large inter- and intra-domain variations. The code, pre-trained models, and dataset are available at clovaai/stargan-v2.*

<img src="assets/831_pressure.png"  style="zoom: 50%;" />
<img src="assets/831_temperature.png"  style="zoom: 50%;" />
<img src="assets/831_velocity.png"  style="zoom: 50%;" />

## Software installation
Clone this repository:

```bash
git clone https://github.com/jungletada/CDF-StarGAN.git
cd CDF-StarGAN/
```

Install the dependencies:
```bash
conda create -n cfd python=3.6.7
conda activate cfd
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
```

## Datasets and pre-trained networks
<!-- We provide a script to download datasets used in StarGAN v2 and the corresponding pre-trained networks. The datasets and network checkpoints will be downloaded and stored in the `data` and `expr/checkpoints` directories, respectively. -->


## Evaluation metrics
To evaluate StarGAN v2 using [Fr&eacute;chet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) and [Learned Perceptual Image Patch Similarity (LPIPS)](https://arxiv.org/abs/1801.03924), run the following commands:


```bash
python main.py --mode eval --num_domains 4 --w_hpf 1 \
               --resume_iter 100000 \
               --train_img_dir data/case_data1/fluent_data_fig \
               --val_img_dir data/case_data2/fluent_data_fig \
               --checkpoint_dir expr/checkpoints \
               --eval_dir expr/eval
```

Note that the evaluation metrics are calculated using random latent vectors or reference images, both of which are selected by the [seed number](https://github.com/clovaai/stargan-v2/blob/master/main.py#L35). In the paper, we reported the average of values from 10 measurements using different seed numbers. The following table shows the calculated values for both latent-guided and reference-guided synthesis.

| Dataset <img width=50/> | <img width=15/> FID (latent) <img width=15/>  | <img width=10/> LPIPS (latent) <img width=10/> | <img width=5/> FID (reference) <img width=5/> | LPIPS (reference) | <img width=10/> Elapsed time <img width=10/>  |
| :---------- | :------------: | :----: | :-----: | :----: | :----------:|
| `celeba-hq` | 13.73 &pm; 0.06 | 0.4515 &pm; 0.0006  | 23.84  &pm; 0.03 | 0.3880 &pm; 0.0001 | 49min 51s
| `afhq` | 16.18 &pm; 0.15 | 0.4501 &pm; 0.0007 | 19.78 &pm; 0.01 | 0.4315 &pm; 0.0002 | 64min 49s


## Training networks
To train StarGAN v2 from scratch, run the following commands. Generated images and network checkpoints will be stored in the `expr/samples` and `expr/checkpoints` directories, respectively. Training takes about three days on a single Tesla V100 GPU. Please see [here](https://github.com/clovaai/stargan-v2/blob/master/main.py#L86-L179) for training arguments and a description of them. 

```bash
python main.py --mode train --num_domains 4 --w_hpf 1 \
```

<!-- To download the updated dataset, run the following command:

```bash
bash download.sh afhq-v2-dataset
``` -->
<!-- 
<p align="left"><img width="99%" src="assets/afhqv2_teaser2.jpg" /></p> -->


## License
The source code, pre-trained models, and dataset are available under [Creative Commons BY-NC 4.0](https://github.com/clovaai/stargan-v2/blob/master/LICENSE) license by NAVER Corporation. You can **use, copy, tranform and build upon** the material for **non-commercial purposes** as long as you give **appropriate credit** by citing our paper, and indicate if changes were made. 

For business inquiries, please contact clova-jobs@navercorp.com.<br/>	
For technical and other inquires, please contact yunjey.choi@navercorp.com.


## Citation
If you find this work useful for your research, please cite our paper:

```
@inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Acknowledgements
We would like to thank the full-time and visiting Clova AI Research (now NAVER AI Lab) members for their valuable feedback and an early review: especially Seongjoon Oh, Junsuk Choe, Muhammad Ferjad Naeem, and Kyungjune Baek. We also thank Alias-Free GAN authors for their contribution to the updated AFHQ dataset.
