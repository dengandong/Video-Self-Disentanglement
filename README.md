# Video-Self-Disentanglement

This is a tensorflow implementation of video disentanglement via a self-supervised way.

The framework aims at disentangling the motion and content of a video in an unsupervised way; after that, use the extracted motion code to generate frames with similiar motion and different content, we also use the same motion encoder after the generator to extract the motion code of the synthesized frames and expect the motion code to keep unchaged in this generation process, i.e. view the motion as a supervision.

We expect this model to extract reasonable motion representation to facilitate action recognition and frame synthesis.

![image](https://github.com/antony0621/Video-Self-Disentanglement/blob/master/images/Fig_1.jpg)
![image](https://github.com/antony0621/Video-Self-Disentanglement/blob/master/images/Fig_2.jpg)
