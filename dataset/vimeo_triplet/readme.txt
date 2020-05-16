========================================================================================
=== Vimeo-90K Triplets Dataset for Temporal Frame Interpolation
========================================================================================

Vimeo-90K triplets dataset contains 91701 triplets extracted from 15k video clips. Each triplet is a short RGB video sequence that consists of 3 frames with fixed resolution 448x256. This dataset is designed to temporal frame interpolation. All the videos are downloaded from vimeo.com.

========================================================================================
=== Folder structure
========================================================================================

- sequences: This folder stores all 91701 septuplets. It uses a two-level folder structure, where each folder "%05d/%04d" contains a short video sequence consists of 3 frames: im1.png, im2.png, and im3.png.

- tri_trainlist.txt: contains the list of sequences for training.

- tri_testlist.txt: contains the list of sequences for test.

======================================================
=== Citation
======================================================

If you use this dataset in your work, please cite the following work:

@article{xue17toflow,
  author = {Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  title = {Video Enhancement with Task-Oriented Flow},
  journal = {arXiv},
  year = {2017}
}

For questions, please contact Tianfan Xue (tianfan.xue@gmail.com), Baian Chen(baian@mit.edu), Jiajun Wu (jiajunwu@mit.edu), or Donglai Wei(donglai@csail.mit.edu).

For more information, please refers to our project website and github repo:

Project website: http://toflow.csail.mit.edu/
Github repo: https://github.com/anchen1011/toflow

======================================================
=== Disclaimer
======================================================

This dataset is for non-commercial usage only.
