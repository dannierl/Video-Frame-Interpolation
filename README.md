# Video-Frame-Interpolation

**Training Result**

When we trained our model with the images of the same scene, the training loss continues to decrease within 20 epochs. After a following climbing halted around epoch 40, the training loss decreases again with experience and eventually keeps at a point of stability after epoch 100. The validation loss follows the trend of training loss but a gap remains between these two curves.

Regarding to the accuracy, both training and validation have the similar ascending trend although there is a stagnancy and small drop between epochs 20 and 50 before they eventually climb to a stable higher level.



<img src=".\results\loss_on_same_scene.png" alt="image-loss" style="zoom:90%;" /> <img src=".\results\acc_on_same_scene.png" alt="image-acc" style="zoom: 90%;" />

Figure XX: (a) The training loss and validation loss show the training set maybe is small relative to the validation dataset.

​                   (b) The training accuracy and validation accuracy shows a fair good performance



When we trained our model with the images of different scenes, the training loss continues to decrease within the epochs in which the images are from same scenes. But the loss will jump sharply when the epoch switches to the images from different scenes. Such a phenomena is called catastrophic forgetting because the change of the training data is so significant that the model has to forget previous experience to fit the new data. However, the validation loss has a pretty much smooth trace without sharp jumping during its descending. 

The training accuracy steps down in the first about 100 epochs but quickly rebound and stay at a much higher level. Though the validation accuracy climbs to the plateau steadily and gets stable after epoch 110.



<img src=".\results\train_loss_on_diff_scene.png" alt="img" style="zoom:40%;" /> <img src=".\results\val_loss_on_diff_scene.png" alt="img" style="zoom: 40%;" />



<img src=".\results\acc_on_diff_scene.png" alt="image-20200606135119012" style="zoom: 90%;" />



Figure XX: (a) The training loss has sharp jump while the training data change significantly. It indicates that the training set maybe is small relative to the

​                        validation dataset.

​                   (b) The validation loss shows a relative smooth descending trend.

​                   (c) Both of the training accuracy and validation accuracy reach to a good performance while the training accuracy performs poorly at the early

​                        stage.





| #Parameters (million) | Runtime (seconds) |
| --------------------- | ----------------- |
|                       |                   |





**Testing Result**

We tested our model with Vimeo triplet sets and HEVC data. The table below shows the metrics of MSE, PSNR and SSIM we got when testing with 100 Vimeo triplet sets which are respectively from the same scene and different scenes.

| 100 Vimeo Triplet Sets | MSE     | PSNR    | SSIM   |
| ---------------------- | ------- | ------- | ------ |
| **Same Scene**         | 51.3136 | 18.0673 | 0.6097 |
| **Different Scenes**   | 36.9612 | 21.7903 | 0.7966 |





<img src=".\results\origin_same_scene.png" alt="img" style="zoom:90%;" />    <img src=".\results\interpolated_same_scene.png" alt="img" style="zoom:90%;" />

Figure XX  -- same scene



<img src=".\results\origin_diff_scene.png" alt="img" style="zoom:90%;" />  <img src=".\results\interpolated_diff_scene.png" alt="img" style="zoom:90%;" />

Figure XX  -- different scene