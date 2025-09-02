  <h1 align="center">SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion [CVPR'25]</h1>
  <h3 align="center"><a href="https://zju3dv.github.io/sgformer/">🌐Project page</a> | <a href="https://www.arxiv.org/abs/2503.16825">📝Paper</a></h3>
  <p align="center">
    <a href="https://github.com/gxytcrc/"><strong>Xiyue Guo</strong><sup>1</sup></a>
    ·
    <a href="https://github.com/hjr37/"><strong>Jiarui Hu</strong><sup>1</sup></a>
    ·
    <a href="https://github.com/JunjH/"><strong>Junjie Hu</strong><sup>2</sup></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/bao/"><strong>Hujun Bao</strong><sup>1</sup></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/gfzhang/"><strong>Guofeng Zhang</strong><sup>1*</sup></a>
    <br>
    <sup>1 </sup>State Key Lab of CAD&CG, Zhejiang University,
    <sup>2 </sup>Chinese University of Hong Kong, Shenzhen<br>
    <sup>* </sup>Corresponding author.<br>
  </p>


This is the official implementation of <strong>SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion</strong>. SGFormer is the first satellite-ground cooperative SSC framework that achieves state-of-the-art performance in scene semantic completion.

  <a href="">
    <img src="https://github.com/gxytcrc/fictional-succotash/blob/main/pipeline.jpg" alt="SGFormer pipeline" width="100%">
  </a>

# Training
Download the pretrained weight of the satellite backbone [Semantic-KITTI: Google Drive](https://drive.google.com/file/d/1qjv9dLFNdn_fJ9a2MOxBfwMrhhX9E6Y6/view?usp=drive_link)

```shell
python train.py 
```

# Eval
Download the weight of our model [Semantic-KITTI: Google Drive](https://drive.google.com/file/d/1UZ6YnTw26JzWdgxiqbNF8yV3LCB1jBzD/view?usp=drive_link)
```shell
python eval.py 
```
# Visualization Result
  <a href="">
    <img src="https://github.com/gxytcrc/fictional-succotash/blob/main/vis.jpg" alt="SGFormer pipeline" width="100%">
  </a>


# News
- [x] 2025.02.26 --- Our paper has been accepted at CVPR 2025！  
- [x] 2025.04.08 --- We have updated the `README.md` and are preparing to open-source our code！
- [x] 2025.08.25 --- We have update our code 
