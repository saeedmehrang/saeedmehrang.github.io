---
title: "An Activity Recognition Framework Deploying the Random Forest Classifier and A Single Optical Heart Rate Monitoring and Triaxial Accelerometer Wrist-Band"
date: 2018-02-22
tags: ["accelerometer", "activity recognition", "context awareness", "machine learning", "photoplethysmography", "random forest", "wrist-worn sensors"]
author: ["Saeed Mehrang", "Julia Pietilä", "Ilkka Korhonen"]
description: "This paper presents an activity recognition framework using a single optical heart rate monitoring and triaxial accelerometer wrist-band."
summary: "This paper investigates a range of daily life activities and uses a random forest classifier to detect them based on wrist motions and optical heart rate. The highest accuracy was achieved with a forest of 64 trees and 13-s signal segments."
cover:
    image: "paper1_diagram.png"
    alt: "Activity Recognition Framework"
    relative: true
editPost:
    URL: "https://www.mdpi.com/1424-8220/18/2/613"
    Text: "Sensors"

---

---

##### Download

+ [Paper](https://www.mdpi.com/1424-8220/18/2/613)

---

##### Abstract

Wrist-worn sensors have better compliance for activity monitoring compared to hip, waist, ankle or chest positions. However, wrist-worn activity monitoring is challenging due to the wide degree of freedom for the hand movements, as well as similarity of hand movements in different activities such as varying intensities of cycling. To strengthen the ability of wrist-worn sensors in detecting human activities more accurately, motion signals can be complemented by physiological signals such as optical heart rate (HR) based on photoplethysmography. In this paper, an activity monitoring framework using an optical HR sensor and a triaxial wrist-worn accelerometer is presented. We investigated a range of daily life activities including sitting, standing, household activities and stationary cycling with two intensities. A random forest (RF) classifier was exploited to detect these activities based on the wrist motions and optical HR. The highest overall accuracy of 89.6 ± 3.9% was achieved with a forest of a size of 64 trees and 13-s signal segments with 90% overlap. Removing the HR-derived features decreased the classification accuracy of high-intensity cycling by almost 7%, but did not affect the classification accuracies of other activities. A feature reduction utilizing the feature importance scores of RF was also carried out and resulted in a shrunken feature set of only 21 features. The overall accuracy of the classification utilizing the shrunken feature set was 89.4 ± 4.2%, which is almost equivalent to the above-mentioned peak overall accuracy.

---

##### Citation

Mehrang, Saeed, Julia Pietilä, and Ilkka Korhonen. 2018. "An Activity Recognition Framework Deploying the Random Forest Classifier and A Single Optical Heart Rate Monitoring and Triaxial Accelerometer Wrist-Band." *Sensors* 18 (2): 613. https://doi.org/10.3390/s18020613.

```BibTeX
@article{mehrang2018activity,
  title={An activity recognition framework deploying the random forest classifier and a single optical heart rate monitoring and triaxial accelerometer wrist-band},
  author={Mehrang, Saeed and Pietil{"a}, Julia and Korhonen, Ilkka},
  journal={Sensors},
  volume={18},
  number={2},
  pages={613},
  year={2018},
  publisher={MDPI}
}
```

---

```
