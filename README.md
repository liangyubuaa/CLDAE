# CLDAE

This is an implementation of the DBET method, described in the following paper:

**CLDAE: A Two Stage EEG-based Emotion Recognition Framework Combining Contrastive Learning and Dual-Attention Encoder**

![architecture](./architecture.jpg)

# Abstract

Electroencephalogram (EEG)-based emotion recognition systems face a persistent challenge: achieving robust performance in both cross-subject generalization and within-subject personalization simultaneously. Existing approaches often prioritize one paradigm while compromising the other cross-subject methods struggle to adapt to individual-specific neural signatures, whereas within-subject models require extensive personalized data. To bridge this gap, we propose CLDAE, a contrastive learning enhanced framework that integrates cross-subject EEG signal recombination with a dual-attention feature extraction mechanism.The framework operates in two stages: a contrastive learning pre-training stage and an emotion recognition fine-tuning stage. During pre-training, we employ a data augmentation method that combines EEG signals from different subjects to create new training samples, thereby increasing the accuracy of Cross-Subject and Within-Subject recognition.
A Dual-Attention Encoder, incorporating both temporal and channel attention mechanisms, is used to extract salient features from the augmented EEG data, capturing both time-domain and frequency-domain information enabling joint modeling of population-level invariants and individual discriminative features. The model is then fine-tuned on a labeled emotion dataset for recognition. Experiments on two public datasets the DEAP and SEED-IV and a private dataset MAN dataset demonstrate that CLDAE achieves competitive both within-subject and cross-subject emotion recognition accuracy (e.g.,94.78\% within-subject accuracy, 75.29\% cross-subject accuracy on MAN), outperforming baseline methods. These results validate that our framework is effective for dealing with both Within-Subject and Cross-Subject Emotion Recognition. Our code is available at https://github.com/liangyubuaa/CLDAE

## Requirements

- python 3.8
- For dependencies，see [requirements.txt](./requirements.txt)
## Reference

```
@ARTICLE{11417405,
  author={Cao, Rongqi and He, Jian and Liang, Yu and Hu, Xiyuan and Peng, Tianhao and Wu, Wenjun and Niu, Shuang and Mumtaz, Shahid},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={CLDAE: A Two Stage EEG-based Emotion Recognition Framework Combining Contrastive Learning and Dual-Attention Encoder}, 
  year={2026},
  volume={},
  number={},
  pages={1-13},
  keywords={Electroencephalography;Brain modeling;Emotion recognition;Feature extraction;Adaptation models;Contrastive learning;Transformers;Data models;Accuracy;Training;EEG;Emotion Recognition;Contrastive Learning;Data Augmentation;Dual-attention encoder},
  doi={10.1109/JBHI.2026.3668381}}
```
