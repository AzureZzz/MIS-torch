

# MIS-TORCH

***

A segmented network code architecture based on Pytorch，including Unet、Unet++、ResUnet。

## Environment

***

- Pytorch：1.6.0+cu101 +
- torchversion：0.7.0+cu101 +
- Python：3.7+
- Other：tensorboard、tqdm、opencv-python、PIL，etc.

## How to run

***

Set parameters by **get_args()** in main.py

Run **main.py**

## Watch training results

***

1. Open the terminal in project path
2. Execute the command：**tensorboard --logdir runs**
3. Access to web sites：http://localhost:6006/ 

<center>     
    <img src=".\readme_imgs\tensorboard.png">
</center>

## DataSets

***

- our_min

```
超声单目标小图数据集：1088张（不含I级样本）
链接：https://pan.baidu.com/s/19DfJ-9SlYJWbRIWTTfE0og 
提取码：qdp2 
```

<center>     
    <img height="256" width="256"  src=".\readme_imgs\our_min_example.png">
    <img height="256" width="256"  src=".\readme_imgs\our_min_example_mask.png">
</center>

- our_large

```
超声整体大图数据集：985张（不含I级样本）
链接：https://pan.baidu.com/s/1QHO1kmV5K_q6sEYTPcavWQ 
提取码：wtor 
```

<center>     
    <img height="256" width="256" src=".\readme_imgs\our_large_example.png">
    <img height="256" width="256" src=".\readme_imgs\our_large_example_mask.png">
</center>

- corneal

```
链接：https://pan.baidu.com/s/1LV2LBGqr0_Ewb1p_7DBdcQ 
提取码：mkn4 
```

<center>     
    <img height="256" width="256" src="readme_imgs\corneal_example.png">
    <img height="256" width="256" src="readme_imgs\corneal_example_mask.png">
</center>


- drive_eye

```
链接：https://pan.baidu.com/s/1q2-QoNjZUC1yhyilXSPPMQ 
提取码：ushx 
```

<center>     
    <img height="256" width="256" src="readme_imgs\drive_eye_example.png">
    <img height="256" width="256" src="readme_imgs\drive_eye_example_mask.png">
</center>


- dsb2018_cell

```
链接：https://pan.baidu.com/s/1k1geEWsFjDiZCU9L_JVBVA 
提取码：zr7r 
```

<center>     
    <img height="256" width="256" src=".\readme_imgs\dsb2018_cell_example.png">
    <img height="256" width="256" src=".\readme_imgs\dsb2018_cell_example_mask.png">
</center>

- esophagus

```
链接：https://pan.baidu.com/s/1FG7Ch9i9tALGyUkIckl8aw 
提取码：95jp 
```

<center>     
    <img height="256" width="256" src=".\readme_imgs\esophagus_example.png">
    <img height="256" width="256" src=".\readme_imgs\esophagus_example_mask.png">
</center>

- ISBI_cell

```
链接：https://pan.baidu.com/s/1i-jo5_JFD8pYlfK5zQ6hjA 
提取码：jb5m 
```

<center>     
    <img height="256" width="256" src=".\readme_imgs\ISBI_cell_example.png">
    <img height="256" width="256" src=".\readme_imgs\ISBI_cell_example_mask.png">
</center>

- kaggle_lung

```
链接：https://pan.baidu.com/s/153egsyE1ThzUE2iJaGCSLw 
提取码：5ttp 
```

<center>     
    <img height="256" width="256" src=".\readme_imgs\kaggle_lung_example.png">
    <img height="256" width="256" src=".\readme_imgs\kaggle_lung_example_mask.png">
</center>

- liver

```
链接：https://pan.baidu.com/s/1xV-WW88YyOaRE397MyPapw 
提取码：94ly 
```

<center>     
    <img height="256" width="256" src=".\readme_imgs\liver_example.png">
    <img height="256" width="256" src=".\readme_imgs\liver_example_mask.png">
</center>

- TN-SCUI

```
链接：https://pan.baidu.com/s/1EAkK8rDjD2EVAcVBl4RZAg 
提取码：gpli 
注：MICCAI Workshop比赛数据集，不可用于学术及商业用途
```

<center>     
    <img height="256" width="256" src=".\readme_imgs\TN-SCUI_example.png">
    <img height="256" width="256" src=".\readme_imgs\TN-SCUI_example_mask.png">
</center>



