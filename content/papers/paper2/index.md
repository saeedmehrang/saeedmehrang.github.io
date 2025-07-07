---
title: "End-to-end sensor fusion and classification of atrial fibrillation using deep neural networks and smartphone mechanocardiography"
date: 2022-05-25
tags: ["seimocardiography", "gyrocardiography", "accelerometer", "gyroscope", "atrial fibrillation", "deep learning", "sensor fusion"]
author: ["Saeed Mehrang", "Mojtaba Jafari Tadi", "Timo Knuutila", "Jussi Jaakkola", "Samuli Jaakkola", "Tuomas Kiviniemi", "Tuija Vasankari", "Juhani Airaksinen", "Tero Koivisto", "Mikko Pänkäälä"]
description: "This paper presents an end-to-end deep learning framework for detecting atrial fibrillation (AFib) using smartphone mechanocardiography."
summary: "This paper presents a deep learning framework for detecting atrial fibrillation (AFib) by analyzing the heart’s mechanical functioning using smartphone mechanocardiography. The model achieves high accuracy in classifying sinus rhythm, AFib, and Noise categories."
cover:
    image: "paper2_cover.png"
    alt: "End-to-end sensor fusion and classification of atrial fibrillation"
    relative: true
editPost:
    URL: "https://doi.org/10.1088/1361-6579/ac66ba"
    Text: "Physiological Measurement"

---

---

##### Download

+ [Paper](https://www.theseus.fi/bitstream/handle/10024/788621/Jafaritadi_et_al_Endtoend_2022.pdf?sequence=1)

---

##### Abstract

Objective. The purpose of this research is to develop a new deep learning framework for detecting atrial fibrillation (AFib), one of the most common heart arrhythmias, by analyzing the heart’s mechanical functioning as reflected in seismocardiography (SCG) and gyrocardiography (GCG)signals. Jointly, SCG and GCG constitute the concept of mechanocardiography (MCG), a method used to measure precordial vibrations with the built-in inertial sensors of smartphones. Approach. We present a modified deep residual neural network model for the classification of sinus rhythm, AFib, and Noise categories from tri-axial SCG and GCG data derived from smartphones. In the model presented, pre-processing including automated early sensor fusion and spatial feature extraction are carried out using attention-based convolutional and residual blocks. Additionally, we use bidirectional long short-term memory layers on top of fully-connected layers to extract both spatial and spatiotemporal features of the multidimensional SCG and GCG signals. The dataset consisted of 728 short measurements recorded from 300 patients. Further, the measurements were divided into disjoint training, validation, and test sets, respectively, of 481 measurements, 140 measurements, and 107 measurements. Prior to ingestion by the model, measurements were split into 10 s segments with 75 percent overlap, pre-processed, and augmented. Main results. On the unseen test set, the model delivered average microand macro-F1-score of 0.88 (0.87–0.89; 95% CI) and 0.83 (0.83–0.84; 95% CI)for the segment-wise classification as well as 0.95 (0.94–0.96; 95% CI) and 0.95 (0.94–0.96; 95% CI)for the measurement-wise classification, respectively. Significance. Our method not only can effectively fuse SCG and GCG signals but also can identify heart rhythms and abnormalities in the MCG signals with remarkable accuracy.

---

##### Citation

Mehrang, S., Jafari Tadi, M., Knuutila, T., Jaakkola, J.,Jaakkola, S., Kiviniemi, T., Vasankari, T., Airaksinen, J., Koivisto, T., Pänkäälä, M. 2022. End-to-end sensor fusion and classification of atrial fibrillation using deep neural networks and smartphone mechanocardiography. Physiological Measurement 43(5): 055004.

```BibTeX
@article{mehrang2022end,
  title={End-to-end sensor fusion and classification of atrial fibrillation using deep neural networks and smartphone mechanocardiography},
  author={Mehrang, Saeed and Tadi, Mojtaba Jafari and Knuutila, Timo and Jaakkola, Jussi and Jaakkola, Samuli and Kiviniemi, Tuomas and Vasankari, Tuija and Airaksinen, Juhani and Koivisto, Tero and P{"a}nk{"a}{"a}l{"a}, Mikko},
  journal={Physiological Measurement},
  volume={43},
  number={5},
  pages={055004},
  year={2022},
  publisher={IOP Publishing}
}
```

---
