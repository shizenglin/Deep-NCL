# Crowd-Counting-with-Deep-Negative-Correlation-Learning
The codes for CVPR-2018  paper "Crowd Counting with Deep Negative Correlation Learning" in http://openaccess.thecvf.com/content_cvpr_2018/papers/Shi_Crowd_Counting_With_CVPR_2018_paper.pdf

To run this codes, what you need to do is as follows:

1. Compiling the Caffe codes.
You should have installed Caffe correctly. Then you can clone or download our codes and make some changes in Makefile.config to compile correctly.

2. Preparing your data.
The codes for preparing crowd counting dataset can be found in ¨examples/crowd/shanghaiA/predata¨. If you would like to run your own tasks, you have to write the codes by yourself.

3. Training.
In ¨examples/crowd¨, you can find the network prototxt and solver prototxt, and you should make some changes according to your tasks. If you want to use different ¨K¨ (K stands for the number of baseregresors in the ensemble, we found K=64 gives the best performancein our tasks.), the ¨examples/crowd/shanghaiA/create_prototxt.py¨ can help to generate your network prototxt easily.

4. Testing.
If you use MAE and MSE as your evaluation metrics, you can monitor the testing results in training. You just need to add 
¨layer {
   name: "mae"
   type: "MAELoss"
   bottom: "avgscore"
   bottom: "label"
   top: "mae"
   include {
    phase: TEST
  }
}¨ and 
¨layer {
   name: "mse"
   type: "MSELoss"
   bottom: "avgscore"
   bottom: "label"
   top: "mse"
   include {
    phase: TEST
  }
}¨
to your network prototxt.

5. Please carefully tune learning rate, we found it had a great influence to the training results in our tasks.

6. If you use this codes, please kindly cite our paper:

@InProceedings{Shi_2018_CVPR,

author = {Shi, Zenglin and Zhang, Le and Liu, Yun and Cao, Xiaofeng and Ye, Yangdong and Cheng, Ming-Ming and Zheng, Guoyan},

title = {Crowd Counting With Deep Negative Correlation Learning},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

month = {June},

year = {2018}

}

Please feel free to contact us if you still have any questions.
Zenglin Shi: iezlshi@gmail.com
Le Zhang: zhang.le@adsc.com.sg 


