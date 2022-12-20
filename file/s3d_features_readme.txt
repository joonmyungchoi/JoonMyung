This file is a readme of the spatial pooled Mixed_5c S3D features HowTo100M features trained without labels from https://arxiv.org/abs/1912.06430.
There are not the same ResNet and ResNeXt features used in the original HowTo100M papers.

You can download the zip file (>800GB) at: http://howto100m.inria.fr:6885/howto100m_s3d_features/howto100m_s3d_features.zip

The zip file contains numpy arrays files for each video from HowTo100M.
The features were extracted at 16 fps in contiguous chunks of 16 frames at 224x224, which means that there are exactly one feature per second.
The dimensionality of the features is 1024.
So if a video duration is 1m20s, the feature array would be of shape 80x1024.

You can download the model used for extracting the features here:
PyTorch: https://github.com/antoine77340/S3D_HowTo100M
Tensorflow: https://tfhub.dev/deepmind/mil-nce/s3d/1
