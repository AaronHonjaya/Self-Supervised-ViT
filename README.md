# Self-Supervised Vision Transformer

This is my attempt at creating a Self-Supervised model. I decided to try and make it a general purpose, pretrained model. 

The method I went for was using Masked Autoencoding (MAE), based on the paper "Masked Autoencoders are scalable vision learners" (citied below). The general structure is the same as the MAE paper, but I introduce the Mask Tokens before the encoder, rather than after. To summarize the overall architecture
 1. patchify the images
 2. mask 75% of the patches, repalcing with learnable mask tokens
 3. Pass the sequence to the encoder
 4. Pass the encoder output to the decoder
 5. Regenerate the image and use MSE loss to optimize. 


<img width="500" alt="image" src="https://github.com/user-attachments/assets/df9d888d-e844-4c8e-9f60-22bf0479445b">

The encoder architecture is fairly simple. The masked inputs are projected into the embedding space. Then learnable mask tokens are selected and inserted from a set of learnable tokens (purple patches). Positional embedding is also added to each patch (symbolized by green lines). This is then passed into the transformer which sends it to an MLP head and then outputs a sequence. 


<img width="970" alt="image" src="https://github.com/user-attachments/assets/e4f86231-dbaa-4546-bb6c-5014cdc7bab2">

The Decoder architecture is a copy of the encoder architecture, except positional encoding and mask tokens are not injected. Its just a transformer with an MLP head.

The transformer architecture is the standard transformer architecture (image snipped from Wikipedia: https://en.wikipedia.org/wiki/Vision_transformer)

<img width="478" alt="image" src="https://github.com/user-attachments/assets/058256dd-018b-4017-9731-ee232c7d9e57">



Currently, I will be training on Tiny-ImageNet (cited at bottom). Note that because the test images have no annotations, the val images will be used for testing instead. 

Download: http://cs231n.stanford.edu/tiny-imagenet-200.zip




# Citations

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). Masked autoencoders are scalable vision learners. arXiv. https://arxiv.org/abs/2111.06377

Le, Y., & Yang, X. (2015). Tiny imagenet visual recognition challenge. CS 231N, 7(7), 3.

https://cs231n.stanford.edu/2021/schedule.html (used variety of their lecture slides for reference)



