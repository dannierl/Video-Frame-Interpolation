# Initial Proposal

> Please remember to rename each subtitle 

## What to do/brief introduction: research on Frame Interpolation and a demo - Lin 
    - description
    - diagram  

## How it is related to Deep Learning for CV ...  - Qiming 

With the continuous success of deep learning in the computer field, researchers have tried to combine deep learning with video insertion technology to meet the needs of insertion. Video super-resolution technology is more complicated, not only needs to generate a detailed frame image but also maintain the continuity between images. The principle of this work is to add additional picture frames between two adjacent frames.

To eliminate the frustration in the picture, intelligently generate interpolated frames, and reproduce 24 frames/second or 25 frames/second video to 60 frames/ second or 90 frames/second video. Frequency interpolation frame technology refers to the use of related information between adjacent frames in the video.

Given the input frame at two moments. First, estimate the optical flow and depth map, and then use the proposed depth-aware flow projection layer to generate the intermediate flow. Then, the model warps the input frame, depth map, and context features based on optical flow and local interpolation kernel to synthesize the output frame.



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