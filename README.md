# FSGM:FSGM-Sum: Factorized Spatio-Temporal Gating Mixer for Efficient Video Summarization 


# Requirements
|Ubuntu|GPU|CUDA|cuDNN|conda|python|
|:---:|:---:|:---:|:---:|:---:|:---:|
|20.04.6 LTS|NVIDIA GeForce RTX 4090|12.1|8902|4.9.2|3.8.5|

|h5py|numpy|scipy|torch|torchvision|tqdm|
|:---:|:---:|:---:|:---:|:---:|:---:|
|3.1.0|1.19.5|1.5.2|2.2.1|0.17.1|4.61.0|

```
conda create -n FSGM python=3.8.5
conda activate FSGM
git clone https://github.com/SMRaza1009/FSGM-Sum.git
cd FSGM
pip install -r requirements.txt
```

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Data
 -  You can download it [here (PGL-SUM).](https://github.com/e-apostolidis/PGL-SUM) <br/> <br/>
H5py format of two benchmark video summarization preprocessed datasets (SumMe, TVSum). <br/>
You should download datasets and put them in ```data/``` directory. <br/>
The structure of the directory must be like below. <br/>
```
 ├── data
     └── eccv16_dataset_summe_google_pool5.h5
     └── eccv16_dataset_tvsum_google_pool5.h5
```
You can see the details of both datasets below. <br/>

[SumMe](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_33) <br/>
[TVSum](https://openaccess.thecvf.com/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf) <br/>
<br/>
[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Pre-trained models
It can be implemented by using the same seed number below (123456). <br/>


You can download our pre-trained weights of CSTA. <br/>
There are 5 weights for the SumMe dataset and the other 5 for the TVSum dataset(1 weight for each split). <br/>
As shown in the paper, we tested everything 10 times (without fixation of seed) but only uploaded a single model as a representative for your convenience. <br/>
The uploaded weight is acquired when the seed is 123456, and the result is almost identical to our paper. <br/>
You should put 5 weights of the SumMe in ```weights/SumMe``` and the other 5 weights of the TVSum in ```weights/TVSum```. <br/>
The structure of the directory must be like below. <br/>
```
 ├── weights
     └── SumMe
         ├── split1.pt
         ├── split2.pt
         ├── split3.pt
         ├── split4.pt
         ├── split5.pt
     └── TVSum
         ├── split1.pt
         ├── split2.pt
         ├── split3.pt
         ├── split4.pt
         ├── split5.pt
```

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Training
You can train the final version of our models by command below. <br/>
```
python train.py
```
Detailed explanations for all configurations will be updated later. <br/>

## You can't reproduce our result perfectly.
As shown in the paper, we tested every experiment 10 times without fixation of the seed, so we can't be sure which seeds export the same results. <br/>
Even though you set the seed 123456, which is the same as our pre-trained models, it may result in different results due to the non-deterministic property of the [Adaptive Average Pooling layer](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms). <br/>
Based on my knowledge, non-deterministic operations produce random results even with the same seed. [You can see details here.](https://pytorch.org/docs/stable/notes/randomness.html) <br/>
However, you can get similar results with the pre-trained models when you set the seed as 123456, so I hope this will be helpful for you. <br/>
<br/>
[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Inference
You can see the final performance of the models by command below. <br/>
```
python inference.py
```
All weight files should be located in the position I said above. <br/>
<br/>


# Generate summary videos
You can generate summary videos using our models. <br/>
You can use either videos from public datasets or custom videos. <br/>
With the code below, you can apply our pre-trained models to raw videos to produce summary videos. <br/>
```
python generate_video.py --input_is_file True or False
    --file_path 'path to input video'
    --dir_path 'directory of input videos'
    --ext 'video file extension'
    --save_path 'path to save summary video'
    --weight_path 'path to loaded weights'

e.g.
1)Using a directory
python generate_video.py --input_is_file False --dir_path './videos' --ext 'mp4' --save_path './summary_videos' --weight_path './weights/SumMe/split4.pt'

2)Using a single video file
python generate_video.py --input_is_file True --file_path './videos/Jumps.mp4' --save_path './summary_videos' --weight_path './weights/SumMe/split4.pt'
```
The explanation of the arguments is as follows. <br/>
If you change the 'ext' argument and input a directory of videos, you must modify the ['fourcc'](https://github.com/thswodnjs3/CSTA/blob/7227ee36a460b0bdc4aa83cb446223779365df45/generate_video.py#L34) variable in the 'produce_video' function within the 'generate_video.py' file. <br/>
Additionally, you must update this when inputting a single video file with different extensions other than 'mp4'.
```
1. input_is_file (bool): True or False
    Indicates whether the input is a file or a directory.
    If this is True, the 'file_path' argument is required.
    If this is False, the 'dir_path' and 'ext' arguments are required.

2. file_path (str) e.g. './SumMe/Jumps.mp4'
    The path of the video file.
    This is only used when 'input_is_file' is True.

3. dir_path (str) e.g. './SumMe'
    The path of the directory where video files are located.
    This is only used when 'input_is_file' is False.

4. ext (str) e.g. 'mp4'
    The file extension of the video files.
    This is only used when 'input_is_file' is False.

5. sample_rate (int) e.g. 15
    The interval between selected frames in a video.
    For example, if the video has 30 fps, it will become 2 fps with a sample_rate of 15.

6. save_path (str) e.g. './summary_videos'
    The path where the summary videos are saved.

7. weight_path (str) e.g. './weights/SumMe/split4.pt'
    The path where the model weights are loaded from.

```

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑
