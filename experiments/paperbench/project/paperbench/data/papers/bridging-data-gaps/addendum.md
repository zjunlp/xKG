# Classifier Training (Section 5.2)

There are certain details missing in the paper on how the classifier were trained. The target classifiers used pre-trained models. For DDPM, the pretrained model used is https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt, while for LDM the pretrained model used is https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt. This is in reference to experimental configuration from Section 5.2.

These pre-trained models were fine-tuned by modifying the last layer to output two classes to classify whether images where coming from the source or the target dataset.
To fine-tune the model the authors used Adam as the optimizer with a learning rate of 1e-4, a batch size of 64, and trained for 300 iterations. This is in reference to experimental configuration from Section 5.2.

# Adaptor Module (Section 4, Algorithm 1)

The adaptor module is composed of a down-pooling layer followed by a normalization layer with 3x3 convolution. Then there is a 4 head attention layer followed by an MLP layer reducing feature size to 8 or 16. Then there is an up-sampling layer with a factor of 4, a normalization layer, and 3x3 convolutions. This is in relation to Section 4, algorithm 1.

# Figures 2b and 2c

In Figure 2b and 2c, the x-axis refers to the time-step of the diffusion process, while the y-axis refers to the sampled values produced by the generative model.

# Hyperparameters for Table 3

For the models used to obtain the results in table 3, the following hyper parameters have been used:

For **DDPM - FFHQ to babies**:  
- learning rate = 5e-6  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 3  
- training iterations = 160

For **DDPM - FFHQ to sunglasses**:  
- learning rate = 5e-5  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 15  
- training iterations = 200

For **DDPM - FFHQ to Raphael**:  
- learning rate = 5e-5  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 10  
- training iterations = 500

For **DDPM - LSUN Church to haunted houses**:  
- learning rate = 5e-5  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 10  
- training iterations = 320

For **DDPM - LSUN Church to landscape drawings**:  
- learning rate = 5e-5  
- C = 16  
- omega = 0.02  
- J = 10  
- Gamma = 10  
- training iterations = 500

For **LDM - FFHQ to babies**:  
- learning rate = 5e-6  
- C = 16  
- omega = 0.02  
- J = 10  
- Gamma = 5  
- training iterations = 320

For **LDM - FFHQ to sunglasses**:  
- learning rate = 1e-5  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 5  
- training iterations = 280

For **LDM - FFHQ to Raphael**:  
- learning rate = 1e-5  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 5  
- training iterations = 320

For **LDM - LSUN Church to haunted houses**:  
- learning rate = 2e-5  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 5  
- training iterations = 500

For **LDM - LSUN Church to landscape drawings**:  
- learning rate = 2e-5  
- C = 8  
- omega = 0.02  
- J = 10  
- Gamma = 5  
- training iterations = 500