# BehavQuant
# (Deep Learning-Based Analysis of Rat Social Interaction)
This pipeline uses YOLOv8 for object detection and BoT-SORT for tracking to estimate rat position and automatically measure behavioral variables. It enables consistent behavior data extraction from raw experimental videos. The dataset used for training was provided by Roboflow (https://universe.roboflow.com/2024-project1/cos_bbox_test).

## Overview

This repository contains the Python-based analysis programs used in the study:

**"The impact of partner interaction on brief social buffering in adolescent female rats as analyzed by deep learning-based object detection algorithms"**

The programs were developed to automatically analyze rat behavior and social interactions from video recordings using deep learning-based object detection and multi-object tracking.

The analysis pipeline includes:

1. Video trimming based on predefined start times
2. Rat detection using YOLOv8
3. Multi-object tracking using BoT-SORT
4. Extraction and storage of tracking data in CSV format
5. Linear interpolation of missing bounding-box coordinates
6. Visualization of detected bounding boxes for verification
7. Calculation of behavioral and social interaction variables
 
<img width="747" height="835" alt="image" src="https://github.com/user-attachments/assets/d6277d42-7f03-4b82-aeec-712be932e25b" />


The YOLOv8 model was trained using 3,101 labeled images extracted from experimental videos. Images were annotated using Roboflow and randomly divided into training (70%), validation (20%), and test (10%) datasets. The pretrained `yolov8l.pt` model was trained for 1,000 epochs with a batch size of 32, and the resulting `best.pt` model was used for subsequent video analysis.

The automated analysis was used to calculate behavioral variables including:

- Black room preference
- Step-through latency
- Number of entries into the black room
- Duration in the same room
- Average distance between subjects
- Duration at distances ≤ 70 mm

For further details regarding the experimental procedures and analysis methods, please refer to the associated publication.

## Reference

If you use this code or analysis pipeline, please cite:

Seo, M., Bae, S.-G., & Noh, J. (2025). The impact of partner interaction on brief social buffering in adolescent female rats as analyzed by deep learning-based object detection algorithms. *Physiology & Behavior, 297*, 114934. https://doi.org/10.1016/j.physbeh.2025.114934

### Software and Algorithms

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* [Computer software]. https://github.com/ultralytics/ultralytics
Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). BoT-SORT: Robust associations multi-pedestrian tracking. *arXiv*. https://doi.org/10.48550/arXiv.2206.14651
Dwyer, B., Nelson, J., Hansen, T., et al. (2024). *Roboflow (Version 1.0)* [Software]. https://roboflow.com

## Contact

For questions, please send an e-mail to me at minhyo331@gmail.com or create an Issue on GitHub.
