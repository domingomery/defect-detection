# defect-detection


Go to [Google Colab](README1.md)


## Defect Detection in Aluminum Castings using Object Detection Methods


In the automotive industry, light-alloy aluminum castings are considered relevant pieces for roadworthiness. In the automated inspection of aluminum castings, X-Ray testing with computer vision is used to detect the defects that are located inside the test object and are thus not detectable to the naked eye. In this work, we evaluate eight state-of-the-art deep object detection methods (based on YOLO, RetinaNet, and EfficientDet) in the detection of aluminum casting defects. We propose a strategy for training that is performed with a low number of defect-free X-ray images of castings with superimposition of simulated defects (avoiding manual annotations). The proposed solution has been simple, effective, and fast. In our experiments, object detector YOLOv5s was trained in only 2.5 hours, and on the testing dataset (with only real defects) the achieved performance was very high (average precision was 0.90 and the F1 factor was 0.91). This method can process 18 X-ray images per second, i.e., this solution can be used in real time inspection to aid human operators. The code and the datasets used in this paper are available on a public repository for future work of the community. It is clear that in the coming years, deep learning-based methods will be more used by the industry of aluminum castings due to their high effectiveness. This paper attempts to make a contribution from academia in this direction.

* Block-diagram of the proposed method 
<img src="https://github.com/domingomery/defect-detection/blob/master/blockdiagram.png" width="600">



## Simulation of ellipsoidal defects

The method generates a 3D ellipoidal (in 3D space) and projects its onto projection plane using a perspective transformation. The ellipsoidal defects are generated randomly. See [implementation](https://github.com/domingomery/defect-detection/tree/master/ellipsoidal-simulation).

* Use of simulated ellipsoidal defects for training purposes 
<img src="https://github.com/domingomery/defect-detection/blob/master/ellipsoidal-simulation/simulation.png" width="600">


## Implementation

In this example, we use series C0001 of [GDXray+](https://domingomery.ing.puc.cl/material/gdxray/). Training and validation using simulated ellipsoidal defects only. Testing on real defects only. Training, Validation and Testing images belong to the same type of wheel. The idea is to demonstrate that we can inspect a type of wheel if we train using the images of this wheel with no defects + simulated defects and test the model on X-ray images of the same type of wheel with real defects.


* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BbFfm8u28USgn2fhR2ybkd4WIVxqPW9I?usp=sharing) RetinaNet

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-8w8WPOc9vTqhHbor9NnqS6Lpujdbe9O?usp=sharing) YOLOv3

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QByPHaz3FhirHeWqV9JF0tuzHLj5MqCv?usp=sharing) YOLOv5

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ltuNKXI7mdk1cp3LTxDjkjv3l06h9P7v?usp=sharing) EfficientDet

See [jupyter notebboks](https://github.com/domingomery/defect-detection/tree/master/object-detectors)

## Results

Results on seven testing images of C0001 (one per column). The first row is the original testing image. The following eight rows are the results using YOLOv3-Tiny, YOLOv3-SPP, YOLOv5s, YOLOv5l, YOLOv5m, YOLOv5x, RetinaNet and EfficientDet respectively (ground truth in red, and detection in green).}

* Achieved results 
<img src="https://github.com/domingomery/defect-detection/blob/master/results.jpg" width="1000">



