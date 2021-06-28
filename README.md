# Camera-Based Table Tennis Posture Analysis
## Demo
![demo](./doc/demo.gif)
## The Problem
Table tennis players need analyses of their opponents’ postures to optimize their game strategies, but it is too laborious and time-consuming to calculate a player’s postures by hands. Besides, most existing models are sensor-based.
## Our Solution
We built a system to classify players’ postures (forehand and backhand) automatically based on their past game and practice videos, and we calculate ratios of players’ postures automatically based on the prediction from those classifiers.
## Methodologies
* Postures Analysis using Machine Learning algorithms
* Semantic Segmentation for Ball Tracking and Table Detection
### Postures Analysis using Machine Learning algorithms
#### Data Collection
We recorded 8 videos of different players from the side of the tables at 30 fps, and the length of the videos are from 20 seconds to 100 seconds. The camera moves slightly during the videos, and has different offsets between different videos.
#### Data Preprocessing
Training on images is not the best approach, because it is costly and time-consuming due to the high dimension. Also, it is easy to be distracted because too many other information are irrelevant to postures.  
Hence, we train models on data containing keypoints of body obtained by OpenPose, which is more efficient on both training costs and time, and also concentrates on players’ body motions.
###### OpenPose is an open source tools authored by CMU Perceptual Computing Lab. It is the first real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints. It is able to handle image and video inputs and can output labeled images, videos, and JSON files that record key points of human body.
![Screen Shot 2021-06-28 at 10 25 54 AM](https://user-images.githubusercontent.com/46513807/123570636-39b72880-d7fb-11eb-8c60-ea9c3ec68b43.png)
#### Models
We did experiments on SVM, CNN, LSTM, and many other models, but CNN and other models performed not well. The accuracies of those models approximated to the baseline and some were even lower than it.  
Finally, we chose SVM, which performed the best both on training and test data, and LSTM, which performed well at training data but test data, as the finalist.
##### LSTM-Based Model
![Screen Shot 2021-06-28 at 10 28 15 AM](https://user-images.githubusercontent.com/46513807/123570795-8dc20d00-d7fb-11eb-8d36-4361b1d3efd7.png)
#### Evaluation
| Model       | Left Model Accuracy | Right Model Accuracy |
|-------------|---------------------|----------------------|
| SVM-RBF     | 89%                 | 75%                  |
| SVM-Sigmoid | 75%                 | 57%                  |
| SVM-Linear  | 82%                 | 95%                  |
| LSTM        | 88%                 | 57%                  |


### Semantic Segmentation for Balls and Tables
#### Data Labeling
![Screen Shot 2021-06-28 at 10 31 07 AM](https://user-images.githubusercontent.com/46513807/123571028-f4dfc180-d7fb-11eb-9db8-66c6a9fe304f.png)
#### Data Augmentation
![Screen Shot 2021-06-28 at 10 31 35 AM](https://user-images.githubusercontent.com/46513807/123571057-045f0a80-d7fc-11eb-9e40-9457fb7310ae.png)
#### EfficientNet
EfficientNet is proposed by Google AI in 2019 and it uses a simple but highly effective compound coefficient to uniformly scales all dimensions of width, depth, and resolution.  
Unlike other models that arbitrary scale a single dimension of the network, the compound scaling method uniformly scales up all dimensions in a principled way.

![Screen Shot 2021-06-28 at 10 33 30 AM](https://user-images.githubusercontent.com/46513807/123571177-48eaa600-d7fc-11eb-94b0-0bdbbbff37be.png)
#### U-Net Architecture
This architecture allows us to use a pre-trained model that has been used for a classification task - on a dataset such as ImageNet - as our encoder. Here, we use EfficientNet as the U-Net’s encoder.

![Screen Shot 2021-06-28 at 10 34 53 AM](https://user-images.githubusercontent.com/46513807/123571288-7a637180-d7fc-11eb-8e8e-dad5318a0335.png)

#### Evaluation
![Screen Shot 2021-06-28 at 10 35 49 AM](https://user-images.githubusercontent.com/46513807/123571347-9bc45d80-d7fc-11eb-85b2-46800ff774f7.png)
#### Results
![Screen Shot 2021-06-28 at 10 36 02 AM](https://user-images.githubusercontent.com/46513807/123571369-a41c9880-d7fc-11eb-98c0-90417b68acff.png)

## Video 
We output videos with the results of the two above mentioned methods.
### Optimization
White points in backgrounds may be detected as balls. To deal with the problem, we recover pixels that be detected as balls at 70% of all the frames in a video.
 
![Screen Shot 2021-06-28 at 10 37 03 AM](https://user-images.githubusercontent.com/46513807/123571436-c7dfde80-d7fc-11eb-9b7f-4e598307880f.png)

## Conclusion
* We developed a table tennis posture analysis system only using camera images. Successfully achieved 89% accuracy on left and 95% accuracy on right
* The semantic segmentation for both balls and tables performs well with 90% average IoU score
* We successfully increased the efficiency of video analyses by automatically calculating the postures distributions and automatically breaking down videos

## References
[1] R. Voeikov, N.Falaleev, R. Baikulov. TTNet: Real-time temporal and spatial video analysis of table tennis. CVPR. 2020.  
[2] C. B. Lin, Z. Dong, W. K. Kuan, Y. F. Huang. A Framework for Fall Detection Based on OpenPose Skeleton and LSTM/GRU Models. In Applied Science. 2020.  
[3] Z. Cao, G. Hidalgo, T. Simon, S. E. Wei, Y. Sheikh. OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol.43, No.1, pp. 172-186, Jan. 1 2021.  
[4] C. Sawant. Human activity recognition with openpose and Long Short-Term Memory on real time images. IEEE 5th International Conference for Convergence in Technology (I2CT). 2020.  
 
