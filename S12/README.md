## Goal

* Do Inference using YOLOV3 model on user image
* Create a custom YoloV3 model on 4 classes, wherein the images are procured, labelled manually and the model files, config and other hyperparameters are accordingly changed as per requirements
* Train the custom model until sufficient mAP is acheieved and infer on test dataset
* Test the model with a video, wherein the video is divided in frames using ffmpeg, infered and stitched back as a video. Upload the same as a video.

### Dataset and Class Info:
The goal is to detect four Jujutsu Kaisen Characters from the anime Jujutsu Kaisen (JJK). The dataset was manually procured using Fatkun batch downloader and the video of the series. The procured images were manually labelled and trained. Many customizations were made, which can be seen in the comments of the Colab File.

### Inferred Video Link using custom model

*Link:* https://youtu.be/9NCLZhgIoZw

### Inferred Images (4 Per class) using custom model
<p float="left">
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/a.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/b.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/c.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/d.jpeg" width="200" height="112" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/e.jpeg" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/f.jpeg" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/g.jpeg" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/h.jpeg" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/i.jpeg" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/j.jpeg" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/k.jpeg" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/k.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/l.jpeg" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/m.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/n.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S12/Images/o.png" width="200" />
</p>


### Image Infered using COCO trained YOLOV3 model
![Infer](https://github.com/gremlin97/EVA-8/blob/main/S12/Images/myy.jpeg)
