# Adversarial Attacks and Defenses in MIA
A curated list of papers concerning Adversarial Medical Image Analysis (AdvMIA).

## Survey
Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges

**Febuary 2025**: 💡The code has been released.

**October 2024**: 💡The survey is now accepted by <em>ACM Computing Surveys</em>.

**March 2023**: We release the initial version of our survey paper by summarizing the current state of the AdvMIA methods with a unified benchmark evaluation of adversarial robustness for medical image analysis [on arXiv](https://arxiv.org/abs/2303.14133)

## Citing

If you find this project useful for your research, please kindly cite our survey paper.

```
@article{dong2024survey,
  title={Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges},
  author={Dong, Junhao and Chen, Junxi and Xie, Xiaohua and Lai, Jianhuang and Chen, Hao},
  journal={ACM Computing Surveys},
  volume={57},
  number={3},
  pages={1--38},
  year={2024},
  publisher={ACM New York, NY}
}
```



## Code
### Training

#### Single Label Classification

Train a Bi- or Multi-classification model by running
```commandline
python trainCls.py --dataset <str> --wd <float> --lr <float> --method <str> --model <str> --helperModel <str> --helperWeight <str> --eps <int> --epoch <int> --bs <int> --gpu <+>
```
| arg          | description                                                                                          |
|--------------|------------------------------------------------------------------------------------------------------|
| dataset      | The dataset for training. Must be in [mesidorBin, mesidorMulti, melBIn, melMulti, xrayBin].          |
| method       | The training method. Must be in [nat, pgdat, trades, mart, hat].                                     |
| model        | The classification model. We include [res18, chexnet, mv2].                                          |
| helperModel  | The helper model required by HAT. Left blank if not using HAT.                                       |
| helperWeight | The path of the helper model's weight. Left blank if not using HAT.                                  |
| eps          | The perturbation budget of adversarial examples for adversarial training, including [0, 1, 2, 4, 8]. |
| epoch        | The number of epochs during the training.                                                            |
| bs           | The batch size during the training.                                                                  |
| gpu          | The GPU ids that are visible during the training.                                                    |
| wd           | The weight decay.                                                                                    |
| lr           | The learning rate.                                                                                   |

#### Multi-Label Classification

ChestX-ray 14 provides multiple labels for each sample. Train a multi-label classification model by running
```commandline
python trainChex.py --wd <float> --lr <float> --method <str> --model <str> --eps <int> --epoch <int> --bs <int> --gpu <+>
```
| arg          | description                                                                                          |
|--------------|------------------------------------------------------------------------------------------------------|
| method       | The dataset for training. Must be in [nat, pgdat].                                                   |
| model        | The classification model. We include [res18, chexnet, mv2].                                          |
| eps          | The perturbation budget of adversarial examples for adversarial training, including [0, 1, 2, 4, 8]. |
| epoch        | The number of epochs during the training.                                                            |
| bs           | The batch size during the training.                                                                  |
| gpu          | The GPU ids that are visible during the training.                                                    |
| wd           | The weight decay.                                                                                    |
| lr           | The learning rate.                                                                                   |

#### Segmentation

Train a segmentation model by running

```commandline
python trainSeg.py --dataset <str> --wd <float> --lr <float> --method <str> --model <str> --eps <int> --epoch <int> --bs <int> --gpu <+>
```
| arg     | description                                                                                         |
|---------|-----------------------------------------------------------------------------------------------------|
| dataset | The training method. Must be in [mel, xray].                                                        |
| method  | The training method. Must be in [nat, pgdat].                                                       |
| model   | The classification model. We include [unet, segnet].                                                |
| eps     | The perturbation budget of adversarial examples for adversarial training, including [0, 1, 2, 4, 8] |
| epoch   | The number of epochs during the training.                                                           |
| bs      | The batch size during the training.                                                                 |
| gpu     | The GPU ids that are visible during the training.                                                   |
| wd      | The weight decay.                                                                                   |
| lr      | The learning rate.                                                                                  |

---

### Evaluation

#### Single Label Classification

Evaluate the Bi- or Multi-classification model by running

```commandline
python testCls.py --dataset <str> --targetModel <str> --targetPath <str> --surrogateModel <str> --surrogatePath <str> --bs <int> --gpu <+>
```

| arg             | description                                                                                                      |
|-----------------|------------------------------------------------------------------------------------------------------------------|
| dataset         | The dataset for evaluation. Must be in [mesidorBin, mesidorMulti, melBIn, melMulti, xrayBin].                    |
| targetModel     | The target model for evaluation. We include [res18, chexnet, mv2].                                               |
| targetWeight    | The path of the target model's weight.                                                                           |
| surrogateModel  | The surrogate model for evaluation. We include [res18, chexnet, mv2]. Left blank if targetModel==surrogateModel. |
| surrogateWeight | The path of the surrogate model's weight. Left blank if targetWeight==surrogateWeight.                           |
| bs              | The batch size during the evaluation                                                                             |
| gpu             | The GPU ids that are visible during the evaluation.                                                              |

#### Multi-Label Classification

Evaluate the multi-label classification model by running

```commandline
python testChex.py --targetModel <str> --targetPath <str> --surrogateModel <str> --surrogatePath <str> --bs <int> --gpu <+>
```

| arg             | description                                                                                                      |
|-----------------|------------------------------------------------------------------------------------------------------------------|
| targetModel     | The target model for evaluation. We include [res18, chexnet, mv2].                                               |
| targetWeight    | The path of the target model's weight.                                                                           |
| surrogateModel  | The surrogate model for evaluation. We include [res18, chexnet, mv2]. Left blank if targetModel==surrogateModel. |
| surrogateWeight | The path of the surrogate model's weight. Left blank if targetWeight==surrogateWeight.                           |
| bs              | The batch size during the evaluation                                                                             |
| gpu             | The GPU ids that are visible during the evaluation.                                                              |


#### Segmentation

Evaluate the segmentation model by running

```commandline
python testSeg.py --dataset <str> --targetModel <str> --targetPath <str> --surrogateModel <str> --surrogatePath <str> --bs <int> --gpu <+>
```

| arg             | description                                                                                                      |
|-----------------|------------------------------------------------------------------------------------------------------------------|
| dataset         | The dataset for evaluation. Must be in [mel, xray].                                                              |
| targetModel     | The target model for evaluation. We include [unet, segnet].                                                      |
| targetWeight    | The path of the target model's weight.                                                                           |
| surrogateModel  | The surrogate model for evaluation. We include [unet, segnet]. Left blank if targetModel==surrogateModel. |
| surrogateWeight | The path of the surrogate model's weight. Left blank if targetWeight==surrogateWeight.                           |
| bs              | The batch size during the evaluation                                                                             |
| gpu             | The GPU ids that are visible during the evaluation.                                                              |

---

### Visualization

To generate the Class Activation Mapping (CAM) of the classification model, run

```commandline
python gradVisual.py --dataset <str> --model <str> --weight <str> --saveDir <str> --bs <int> --gpu <+>
```
| arg             | description                                                                                               |
|-----------------|-----------------------------------------------------------------------------------------------------------|
| dataset         | The dataset for evaluation. Must be in [mel, xray].                                                       |
| model           | The classification model. We include [res18, chexnet, mv2].                                               |
| weight          | The path of the classification model's weight.                                                            |
| bs              | The batch size during the evaluation                                                                      |
| gpu             | The GPU ids that are visible during the evaluation.                                                       |

