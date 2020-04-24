# Initial Proposal

> Please remember to rename each subtitle 

## What to do/brief introduction: research on Frame Interpolation and a demo - Lin 
    - description
    - diagram  

## How it is related to Deep Learning for CV ...  - Qiming 

Super-resolution is based on the concept proposed by the human visual system. The 1981 Nobel Prize winners David Hubel and Torsten Wiesel discovered that the information processing method of the human visual system is hierarchical. The first layer is the original data input. When a person sees a face image, they will first recognize the edges such as points and lines. Then enter the second layer, it will recognize some basic components in the image, such as eyes, ears, nose. Finally, an object model is generated, which is a complete face.

Convolutional neural network (CNN) in deep learning imitates the processing of the human visual system. Because of this, computer vision is one of the best application areas for deep learning. Super-resolution is a classic application in computer vision. Super-resolution is a method to improve image resolution through software or hardware methods. Its core idea is to exchange temporal bandwidth for spatial resolution. To put it simply, when I can't get an ultra-high-resolution image, I can take a few more images and then combine this series of low-resolution images into a high-resolution image. This process is called super-resolution reconstruction.

Video super-resolution technology is more complicated, not only needs to generate a detailed frame image but also maintain the continuity between images. To eliminate the frustration in the picture, intelligently generate interpolated frames, and reproduce 24 frames/second or 25 frames/second video to 60 frames/ second or 90 frames/second video. Most existing methods are supervised learning. For an original image and a target image, the mapping relationship between them is learned to obtain an enhanced image. However, such data sets are relatively few, and many are artificially adjusted, so self-supervised or weakly supervised methods are needed to solve this problem.

//not sure about mentioning self-supervised or weakly supervised methods
//besides CNN, what else needs to be mentioned in this section?


## Steps: - Daniel 
    - dataset 
    - research paper 
    - env/demo (implementation)  

## Schedule - Daniel  

## Results and associalted emtrics - Wang
    - results: 24/25 -> 60, -> 90
    - compared with other works  

## Risk - Wang  