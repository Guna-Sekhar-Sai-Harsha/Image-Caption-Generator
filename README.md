## Image Caption Generator

A neural network to generate captions for an image using CNN and LSTM & GRU with BEAM Search.

<p align="center">
  <strong>Examples</strong>
</p>

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*6BFOIdSHlk24Z3DFEakvnQ.png" width="85%" title="Example of Image Captioning" alt="Example of Image Captioning">
</p>

<p align="center">
	Image Credits : <a href="https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2">Towardsdatascience</a>
</p>

## Table of Contents

1. [Requirements](#1-requirements)
2. [Training parameters and results](#2-training-parameters-and-results)
3. [Generated Captions on Test Images](#3-generated-captions-on-test-images)
4. [Procedure to Train Model](#4-procedure-to-train-model)
5. [Procedure to Test on new images](#5-procedure-to-test-on-new-images)
6. [Configurations  (config cell in the notebook)](#6-Configurations (#6-configurations-(config-cell-in-the-notebook))
7. [Frequently encountered problems](#7-frequently-encountered-problems)
8. [TODO](#8-todo)
9. [References](#9-references)

## 1. Requirements

Recommended System Requirements to train model.

<ul type="square">
	<li>A good CPU and a GPU with atleast 8GB memory</li>
	<li>Atleast 8GB of RAM</li>
	<li>Active internet connection so that keras can download inceptionv3/vgg16 model weights</li>
</ul>

Required libraries for Python along with their version numbers used while making & testing of this project

<ul type="square">
	<li>Python - 3.6.7</li>
	<li>Numpy - 1.16.4</li>
	<li>Tensorflow - 1.13.1</li>
	<li>Keras - 2.2.4</li>
	<li>nltk - 3.2.5</li>
	<li>PIL - 4.3.0</li>
	<li>Matplotlib - 3.0.3</li>
	<li>tqdm - 4.28.1</li>
</ul>

<strong>Flickr8k Dataset:
<ul type="square">
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip">Flickr8k_Dataset</a></li>
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip">Flickr8k_text</a></li>
	Download Link Credits:<a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/"> Jason Brownlee</a>
</ul>

<strong>Important:</strong> After downloading the dataset, put the reqired files in train_val_data folder

## 2. Training parameters and results

#### NOTE

- `batch_size=64` took ~14GB GPU memory in case of *InceptionV3 + AlternativeLSTM* and *VGG16 + AlternativeLSTM*
- `batch_size=64` took ~8GB GPU memory in case of *InceptionV3 + LSTM* and *VGG16 + LSTM*
- `batch_size=64` took ~8GB GPU memory in case of *InceptionV3 + GRU* and *VGG16 + GRU*

- **If you're low on memory**, use google colab or reduce batch size
- In case of BEAM Search, `loss` and `val_loss` are same as in case of argmax since the model is same

| Model & Config | Argmax | BEAM Search |
| :--- | :--- | :--- |
| **InceptionV3 + AlternativeLSTM** <ul><li>Epochs = 20</li><li>Batch Size = 64</li><li>Optimizer = Adam</li></ul> |<ul>**Crossentropy loss**<br>*(Lower the better)*<li>loss(train_loss): 2.31</li><li>val_loss: 3.1</li>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.60</li><li>BLEU-2: 0.35</li><li>BLEU-3: 0.24</li><li>BLEU-4: 0.12</li></ul> |<ul>**k = 3**<br><br>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.60</li><li>BLEU-2: 0.35</li><li>BLEU-3: 0.24</li> <li>BLEU-4: 0.12</li></ul> |
| **InceptionV3 + LSTM** <ul><li>Epochs = 11</li><li>Batch Size = 64</li><li>Optimizer = Adam</li></ul> |<ul>**Crossentropy loss**<br>*(Lower the better)*<li>loss(train_loss): 2.12</li><li>val_loss: 3.39</li>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.58</li><li>BLEU-2: 0.33</li><li>BLEU-3: 0.23</li><li>BLEU-4: 0.11</li></ul> |<ul>**k = 3**<br><br>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.59</li><li>BLEU-2: 0.34</li><li>BLEU-3: 0.24</li><li>BLEU-4: 0.12</li></ul> |
| **InceptionV3 + GRU** <ul><li>Epochs = 9</li><li>Batch Size = 64</li><li>Optimizer = Adam</li></ul> |<ul>**Crossentropy loss**<br>*(Lower the better)*<li>loss(train_loss): 2.18</li><li>val_loss: 3.36</li>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.60</li><li>BLEU-2: 0.35</li><li>BLEU-3: 0.24</li><li>BLEU-4: 0.12</li></ul> |<ul>**k = 3**<br><br>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.60</li><li>BLEU-2: 0.35</li><li>BLEU-3: 0.25</li><li>BLEU-4: 0.13</li></ul> |
| **VGG16 + AlternativeLSTM** <ul><li>Epochs = 18</li><li>Batch Size = 64</li><li>Optimizer = Adam</li></ul> |<ul>**Crossentropy loss**<br>*(Lower the better)*<li>loss(train_loss): 2.13</li><li>val_loss: 3.21</li>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.57</li><li>BLEU-2: 0.33</li><li>BLEU-3: 0.23</li><li>BLEU-4: 0.11</li></ul> |<ul>**k = 3**<br><br>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.58</li><li>BLEU-2: 0.33</li><li>BLEU-3: 0.23</li><li>BLEU-4: 0.11</li></ul> |
| **VGG16 + LSTM** <ul><li>Epochs = 10</li><li>Batch Size = 64</li><li>Optimizer = Adam</li></ul> |<ul>**Crossentropy loss**<br>*(Lower the better)*<li>loss(train_loss): 1.97</li><li>val_loss: 3.65</li>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.57</li><li>BLEU-2: 0.33</li><li>BLEU-3: 0.23</li><li>BLEU-4: 0.11</li></ul> |<ul>**k = 3**<br><br>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.58</li><li>BLEU-2: 0.33</li><li>BLEU-3: 0.23</li><li>BLEU-4: 0.11</li></ul> |
| **VGG16 + GRU** <ul><li>Epochs = 7</li><li>Batch Size = 64</li><li>Optimizer = Adam</li></ul> |<ul>**Crossentropy loss**<br>*(Lower the better)*<li>loss(train_loss): 2.02</li><li>val_loss: 3.62</li>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.58</li><li>BLEU-2: 0.34</li><li>BLEU-3: 0.24</li><li>BLEU-4: 0.12</li></ul> |<ul>**k = 3**<br><br>**BLEU Scores on Validation data**<br>*(Higher the better)*<li>BLEU-1: 0.59</li><li>BLEU-2: 0.35</li><li>BLEU-3: 0.24</li><li>BLEU-4: 0.12</li></ul> |
	
## 3. Generated Captions on Test Images

**Model used** - *InceptionV3 + AlternativeRNN*

| Image | Caption |
| :---: | :--- |
| <img width="50%" src="https://github.com/dabasajay/Image-Caption-Generator/raw/master/test_data/bikestunt.jpg" alt="Image 1"> | <ul><li><strong>Argmax:</strong> A man in a blue shirt is riding a bike on a dirt path.</li><li><strong>BEAM Search, k=3:</strong> A man is riding a bicycle on a dirt path.</li></ul>|
| <img src="https://github.com/dabasajay/Image-Caption-Generator/raw/master/test_data/surfing.jpeg" alt="Image 2"> | <ul><li><strong>Argmax:</strong> A man in a red kayak is riding down a waterfall.</li><li><strong>BEAM Search, k=3:</strong> A man on a surfboard is riding a wave.</li></ul>|

## 4. Procedure to Train Model

1. Clone the repository to preserve directory structure.<br>
`https://github.com/Guna-Sekhar-Sai-Harsha/Image-Caption-Generator`
2. Put the required dataset files in `train_val_data` folder (files mentioned in readme there).
3. Review `configuration cell` for paths and other configurations (explained below).
4. Run `Image_Caption_Generator.ipynb`.

## 5. Procedure to Test on new images

1. Clone the repository to preserve directory structure.<br>
`https://github.com/Guna-Sekhar-Sai-Harsha/Image-Caption-Generator`
2. Train the model to generate required files in `model_data` folder (steps given above).
3. Put the test images in `test_data` folder.
4. Review `configuration cell` for paths and other configurations (explained below).
5. Run `test cell` in `Image_Caption_Generator.ipynb`.

## 6. Configurations (config cell in the notebook)

**config**

1. **`images_path`** :- Folder path containing flickr dataset images
2. `train_data_path` :- .txt file path containing images ids for training
3. `val_data_path` :- .txt file path containing imgage ids for validation
4. `captions_path` :- .txt file path containing captions
5. `tokenizer_path` :- path for saving tokenizer
6. `model_data_path` :- path for saving files related to model
7. **`model_load_path`** :- path for loading trained model
8. **`num_of_epochs`** :- Number of epochs
9. **`max_length`** :- Maximum length of captions. This is set manually after training of model and required for test.py
10. **`batch_size`** :- Batch size for training (larger will consume more GPU & CPU memory)
11. **`beam_search_k`** :- BEAM search parameter which tells the algorithm how many words to consider at a time.
11. `test_data_path` :- Folder path containing images for testing/inference
12. **`model_type`** :- CNN Model type to use -> inceptionv3 or vgg16
13. **`random_seed`** :- Random seed for reproducibility of results

**rnnConfig**

1. **`embedding_size`** :- Embedding size used in Decoder(RNN) Model
2. **`LSTM_units`** :- Number of LSTM units in Decoder(RNN) Model
3. **`dense_units`** :- Number of Dense units in Decoder(RNN) Model
4. **`dropout`** :- Dropout probability used in Dropout layer in Decoder(RNN) Model

## 7. Frequently encountered problems

- **Out of memory issue**:
  - Try reducing `batch_size`
- **Results differ everytime I run script**:
  - Due to stochastic nature of these algoritms, results *may* differ slightly everytime. Even though I did set random seed to make results reproducible, results *may* differ slightly.
- **Results aren't very great using beam search compared to argmax**:
  - Try higher `k` in BEAM search using `beam_search_k` parameter in config. Note that higher `k` will improve results but it'll also increase inference time significantly.

## 8. TODO

- [X] Support for VGG16 Model. Uses InceptionV3 Model by default.

- [X] Implement 2 architectures of RNN Model.

- [X] Support for batch processing in data generator with shuffling.

- [X] Implement BEAM Search.

- [X] Calculate BLEU Scores using BEAM Search.

- [ ] Implement Attention and change model architecture.

- [ ] Support for pre-trained word vectors like word2vec, GloVe etc.

## 9. References

<ul type="square">
	<li><a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf">Show and Tell: A Neural Image Caption Generator</a> - Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan</li>
	<li><a href="https://arxiv.org/abs/1703.09137">Where to put the Image in an Image Caption Generator</a> - Marc Tanti, Albert Gatt, Kenneth P. Camilleri</li>
	<li><a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/">How to Develop a Deep Learning Photo Caption Generator from Scratch</a></li>
</ul>
