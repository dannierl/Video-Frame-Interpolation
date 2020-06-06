# Meeting 4 Notes (May 11)

14 15 16 (17 18 *19 *20 21) 22 23 

12: Lin: 3 temp frame, Qiming finish most by 12 
13: Qiming Test with data; Further plan for Lin and Qiming 
14: Fenyang&Daniel test with data, Daniel: finish model design 
15: -------- 


16: Milestone 1 >>>> Daniel: test pre-pro with Fenyang 


18: Milestone 2 >>>> Test model 


22 23: Milestone X >>>> training

## Dataset 
    - 448 x 256
    - Middlebury dataset
    - 1200 frames (20% validation 60% training 20% testing): one film, one anemie
    - Final dataset: YouTube or Movie (anemie, film)

load_image -> image_to_block -> feed to model -> kernal ->
                                                            -> get pixel -> image ->loss_function
           -> image_to_patch -> concatenate_patch -> 

## Pre-processing 
    - for each pixel we need to find a patch and a block(receptive area)
    - load_image
    - patch 41*41*3 (480-41+1)*(720-41+1): image_to_patch, show_patch, concatenate_patch, 
    - block 82*82*3: image_to_block, show_block

## Model
    - Design model
    - loss_function 
    - metrics: psnr, performance_evaluation(mse, psnr, ssims)

## post-processing 
    - save_image

## Priority 
- load_image - save_image - plot images -Qiming
- find data -Lin

- image_to_block, image_to_patch (Fenyang & *Daniel)
- concatenate_patch 
- show_block, show_patch 

- design model (Daniel & *Fenyang)

## Env
VSCode/PyCharm + Jupyter 

## Interface 
### load_image(path, start, end)
    - path: relative directory string to png dataset folder
    - [start, end) integer: first 100 frames, [0, 100)
    - n_images = end - start
    - return a numpy array with shape: (n_images_set, 3, 720, 480, 3)
```
    - example
    └── project
        ├── helper.py 
        └── dataset
            ├── video0
            └── video1
                ├── frame0
                ├── frame1
                ├── frame2
                └── frame3
                └── frame4
```
    path = "./dateset/video1/"
    start = 0 
    end = 3

### save_image(path, start, image)
    - path: relative directory string to save folder
    - no return 
### plot_image(image) 
    - image (720, 480, 3) numpy array
  
### image_to_block(images, b_size) 
    - (n_image_set, 3, 448, 256, 3) -> (n_image_set, 3, n_block, b_size, b_size, 3) order by Progressive Scan
    - b_size = 82
    - n_block = (448-82+1)*(256-82+1)  
    - return numpy array with shape (n_image_set, 3, n_block, b_size, b_size, 3)
    - note: list[start: end; step], numpy.reshape
### image_to_patch

## concatenate_patch(p1, p2)
    - P1 and P1 with shape(n_patch, 41, 41, 3)
    - return (n_patch, 41, 82, 3)

### show_block(blocks, b_size)  
    - show blocks of one image
    - blocks ((448-82+1)*(256-82+1), b_size, b_size, 3)
    - Method 1: merge then call plot image
    - Plot by progressive scan 
### show_patch

### model(images, b_size)
    - images: a numpy array with shape: (n_images, 3, 720, 480, 3)
    - b_size = 82
