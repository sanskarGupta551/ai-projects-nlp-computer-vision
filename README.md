# 🧠 AI Projects: NLP & Computer Vision

This repository is a curated collection of **AI projects** spanning **Computer Vision**, **Natural Language Processing (NLP)**, and **Multi-Modal AI**. Each project is implemented in a **self-contained Jupyter Notebook** (or script) with explanations, code, and results.

🔗 Useful Links:

* 📂 [GitHub Repo](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision)
* 🤖 [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* 🖼️ [Keras Applications](https://keras.io/api/applications/)
* 📊 [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
* 📸 [Flickr30k Dataset](https://shannon.cs.illinois.edu/DenotationGraph/)

---

## 📊 Project Summary (Skills Matrix)

| **Project**                                                                                            | **Domain**  | **Key Skills / Techniques**                                        | **Tools & Frameworks**     |
| ------------------------------------------------------------------------------------------------------ | ----------- | ------------------------------------------------------------------ | -------------------------- |
| [Face Mask Detection with VGG16](Computer-Vision/Face_Mask_Detection_with_VGG16.ipynb)                 | CV          | Transfer Learning, Binary Classification, Data Augmentation        | TensorFlow, Keras, VGG16   |
| [Facial Emotion Recognition with VGG16](Computer-Vision/Facial_Emotion_Recognition_with_VGG16.ipynb)   | CV          | Multi-class Classification, Emotion Recognition, Transfer Learning | Keras, VGG16               |
| [Fashion MNIST Classification with CNNs](Computer-Vision/Fashion_MNIST_with_CNN%28s%29.ipynb)          | CV          | CNN Architectures, Model Comparison, Feature Visualization         | TensorFlow, Keras          |
| [Green Screening with OpenCV](Computer-Vision/Green_Screening_Images_and_Videos_with_OpenCV.ipynb)     | CV          | Chroma Keying, Real-time Video Processing                          | OpenCV, NumPy              |
| [Image Deblurring with VGG16 + DCGAN](Computer-Vision/Image_Deblurring_with_VGG16.ipynb)               | CV          | GANs, Perceptual Loss, Image Restoration                           | DCGAN, VGG16, TensorFlow   |
| [Image Captioning with Flickr30k](Computer-Vision/Image_Captioning_with_Flickr30k.ipynb)               | CV + NLP    | Encoder-Decoder, Seq2Seq, BLEU Evaluation                          | VGG16, LSTM, Keras         |
| [Tweets Sentiment Analysis (3 Neural Nets)](NLP/Tweets_Sentiment_Analysis_with_3_Neural_Network.ipynb) | NLP         | Sentiment Analysis, DNN/CNN/RNN Comparison, Embeddings             | TensorFlow, Keras, GloVe   |
| [GenZ Tweets Data Pipeline](NLP/GenZ_Tweets_Data_Pipeline_for_Sentiment_Analysis.ipynb)                | NLP         | Text Preprocessing, Regex, Lemmatization, Emoji Normalization      | NLTK, SpaCy, Python        |
| [Next Word Prediction with Bi-LSTM](NLP/Next_Word_Prediction_with_Bidirectional_LSTM.ipynb)            | NLP         | Language Modeling, Sequence Prediction, Perplexity                 | TensorFlow, Keras, Bi-LSTM |
| [Prompt-to-Synopsis Generator](NLP/Prompt_to_Synopsis_Generator_%28Fine-Tuning%29.ipynb)               | NLP         | Fine-Tuning Transformers, Creative Text Generation                 | HuggingFace, GPT-2         |
| [AI Long-Form Story Generator](NLP/Ai_Long_form_Story_Generator_with_Varied_Context.ipynb)             | NLP         | Long-Context Modeling, Story Generation                            | HuggingFace, Transformers  |
| [AI Imagining Stories from Images](Multi-modal/Ai_Imagining_Stories_from_Images.ipynb)                 | Multi-Modal | Image-to-Text, Vision+Language, Storytelling                       | HuggingFace, Transformers  |

---

## 📂 Projects

### 🖼️ Computer Vision

#### 1. [Face Mask Detection with VGG16](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Computer-Vision/Face_Mask_Detection_with_VGG16.ipynb)

* **Situation**: During COVID-19, monitoring mask compliance became critical in public spaces.
* **Task**: Build a system to automatically detect masks from images of people.
* **Action**:

  * Fine-tuned **VGG16 (transfer learning)** pretrained on ImageNet.
  * Applied **data augmentation** (rotation, flipping, zoom) for robustness.
  * Built a binary classifier on labeled mask/no-mask dataset.
* **Result**: Achieved **97% accuracy** on validation data, demonstrating production feasibility for surveillance and healthcare use cases.
* **Tags**: `TensorFlow` · `Keras` · `Transfer Learning` · `CNN` · `Image Classification` · `Model Deployment`
  
  ![Image](https://www.intertecsystems.com/wp-content/uploads/2020/05/face-mask-detection-software-e1591538656411.png)

---

#### 2. [Facial Emotion Recognition with VGG16](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Computer-Vision/Facial_Emotion_Recognition_with_VGG16.ipynb)

* **Situation**: Emotion recognition is important in **mental health monitoring, human-computer interaction, and customer analytics**.
* **Task**: Develop a model to classify facial images into multiple emotions.
* **Action**:

  * Preprocessed **FER-2013 dataset** with grayscale normalization & augmentation.
  * Fine-tuned **VGG16** with added dense layers for 7-class classification.
  * Used **categorical cross-entropy loss** and early stopping.
* **Result**: Reached **72% accuracy**, surpassing traditional ML baselines (e.g., SVMs \~45%).
* **Tags**: `Keras` · `VGG16` · `Image Classification` · `Emotion Recognition` · `Transfer Learning` · `FER-2013`
  
  ![Image](https://i.pinimg.com/originals/b0/bb/1d/b0bb1d0b86bdca1de8ead928064d09d8.png)

---

#### 3. [Fashion MNIST Classification with CNNs](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Computer-Vision/Fashion_MNIST_with_CNN%28s%29.ipynb)

* **Situation**: Fashion MNIST is a standard benchmark for testing deep learning models on real-world classification tasks.
* **Task**: Classify clothing images into 10 categories.
* **Action**:

  * Built multiple **CNN architectures** (2–4 conv layers, pooling, dropout).
  * Compared models using accuracy and loss curves.
  * Visualized feature maps for interpretability.
* **Result**: Achieved **92% accuracy** on test set with deeper CNN.
* **Tags**: `TensorFlow` · `CNN` · `Fashion-MNIST` · `Model Comparison` · `Deep Learning`
  
  ![Image](https://thiagolcmelo.github.io/assets/img/fashion-mnist.png)

---

#### 4. [Green Screening with OpenCV](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Computer-Vision/Green_Screening_Images_and_Videos_with_OpenCV.ipynb)

* **Situation**: **Film, broadcasting, and AR applications** rely on chroma key (green screen).
* **Task**: Replace green backgrounds in images and videos with arbitrary scenes.
* **Action**:

  * Used **OpenCV** to detect and mask green pixel ranges.
  * Replaced with background images/videos dynamically.
  * Implemented for both static images and live video streams.
* **Result**: Delivered real-time background replacement with smooth transitions.
* **Tags**: `OpenCV` · `Computer Vision` · `Chroma Keying` · `Real-Time Video Processing`
  
  ![Image](https://i.ytimg.com/vi/JfyzwWgZT4M/maxresdefault.jpg)

---

#### 5. [Image Deblurring with VGG16 + DCGAN](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Computer-Vision/Image_Deblurring_with_VGG16.ipynb)

* **Situation**: Blurred images affect critical fields like **surveillance and medical imaging**.
* **Task**: Restore sharpness in blurred images.
* **Action**:

  * Built a **DCGAN generator-discriminator architecture**.
  * Used **VGG16 perceptual loss** to guide training.
  * Trained on custom mixed-blur dataset (motion blur, Gaussian blur).
* **Result**: Restored sharper images with **SSIM score improvement of +18% over baseline interpolation**.
* **Tags**: `GAN` · `DCGAN` · `VGG16` · `Image Restoration` · `Perceptual Loss`
  
  ![Image](https://th.bing.com/th/id/R.1c95a17451c44bb937d2275bc6ae14d9?rik=FClK479meeWRPA\&riu=http%3a%2f%2fwww.ece.northwestern.edu%2flocal-apps%2fmatlabhelp%2ftoolbox%2fimages%2fdeblu10a.gif\&ehk=VTBAkwI%2bK%2b2lqG5JtdUoByh24GOJfmcoo8RGX8xAyiY%3d\&risl=\&pid=ImgRaw\&r=0)

---

#### 6. [Image Captioning with Flickr30k](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Computer-Vision/Image_Captioning_with_Flickr30k.ipynb)

* **Situation**: **Image captioning aids accessibility** for the visually impaired and powers multimedia search.
* **Task**: Generate natural language descriptions of images.
* **Action**:

  * Extracted features with **VGG16 encoder**.
  * Trained **LSTM decoder** with teacher forcing on Flickr30k captions.
  * Evaluated with BLEU scores.
* **Result**: Generated fluent captions like *“A boy playing with a dog in the grass”* with **BLEU-4 ≈ 0.41**.
* **Tags**: `VGG16` · `LSTM` · `Seq2Seq` · `Encoder-Decoder` · `Image Captioning` · `Flickr30k`
  
  ![Image](https://petapixel.com/assets/uploads/2016/09/Caption1-800x450.jpg)

---

### 📝 Natural Language Processing (NLP)

#### 7. [Tweets Sentiment Analysis with 3 Neural Networks](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/NLP/Tweets_Sentiment_Analysis_with_3_Neural_Network.ipynb)

* **Situation**: Businesses and political campaigns monitor sentiment on Twitter for decision-making.
* **Task**: Classify tweets into positive, negative, or neutral sentiment.
* **Action**:

  * Built three deep neural network architectures (DNN, CNN, RNN).
  * Preprocessed with regex, stopword removal, embeddings (GloVe).
  * Compared architectures on accuracy/F1.
* **Result**: Best-performing CNN achieved **88% accuracy** on test data.
* **Tags**: `NLP` · `Sentiment Analysis` · `DNN` · `CNN` · `RNN` · `Embeddings`
  
  ![Image](https://www.altexsoft.com/media/2018/09/sentiment_analysis.jpg)

---

#### 8. [GenZ Tweets Data Pipeline for Sentiment Analysis](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/NLP/GenZ_Tweets_Data_Pipeline_for_Sentiment_Analysis.ipynb)

* **Situation**: Raw social media data is noisy with slang, emojis, and hashtags.
* **Task**: Design a reusable pipeline for tweet preprocessing.
* **Action**:

  * Implemented **regex cleaning, tokenization, lemmatization**.
  * Normalized emojis, URLs, and @mentions.
  * Built pipeline both in Jupyter and as a **standalone Python script**.
* **Result**: Produced clean, structured text improving sentiment model accuracy by \~10%.
* **Tags**: `NLP` · `Data Pipeline` · `Regex` · `NLTK` · `SpaCy` · `Preprocessing`
  
  ![Image](https://i.pinimg.com/originals/e1/d0/33/e1d0330eb3bfd698f6c332e0d61f6d11.png)

---

#### 9. [Next Word Prediction with Bi-Directional LSTM](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/NLP/Next_Word_Prediction_with_Bidirectional_LSTM.ipynb)

* **Situation**: Next-word prediction powers **mobile keyboards and search engines**.
* **Task**: Build a language model to predict the next word.
* **Action**:

  * Preprocessed text corpus into n-grams.
  * Trained a **Bi-LSTM sequence model** with embeddings.
  * Evaluated using perplexity and prediction accuracy.
* **Result**: Generated accurate predictions with **perplexity reduced to \~35**, suitable for autocomplete.
* **Tags**: `Bi-LSTM` · `Language Modeling` · `Sequence Prediction` · `Text Generation`
  
  ![Image](https://i.pinimg.com/originals/3f/b4/2e/3fb42e1f042063e0423496529978dd8e.jpg)

---

#### 10. [Prompt-to-Synopsis Generator (Fine-Tuning)](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/NLP/Prompt_to_Synopsis_Generator_%28Fine-Tuning%29.ipynb)

* **Situation**: Entertainment & content industries need tools to expand short prompts into story outlines.
* **Task**: Fine-tune transformer to generate synopses from short prompts.
* **Action**:

  * Fine-tuned **GPT-2 using HuggingFace Transformers**.
  * Applied causal LM loss, LR scheduling, and early stopping.
  * Evaluated coherence & diversity of outputs.
* **Result**: Produced multi-sentence coherent synopses with logical flow from prompts.
* **Tags**: `Transformers` · `GPT-2` · `Fine-Tuning` · `HuggingFace` · `Text Generation`
  
  ![Image]([https://th.bing.com/th/id/R.62fed4c6ba6af08871ed40c89f4d0a44?rik=bHeTpfghWffTSQ\&riu=http%3a%2f%2fsusancushman.com%2fwp-content%2fuploads%2f2012%2f10%2f100_12691.jpg\&ehk=u%2fiowcjKxXZcabvS21Q2ExLG0g6YsIK6vJcgkcE7Xxc%3d\&risl=\&pid=ImgRaw\&r=0](https://i.pinimg.com/originals/08/9e/d4/089ed4e22bcd3e3f28e6327e8387617e.jpg))

---

#### 11. [AI Long-Form Story Generator with Varied Context](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/NLP/Ai_Long_form_Story_Generator_with_Varied_Context.ipynb)

* **Situation**: Longer context improves story coherence but increases complexity.
* **Task**: Build a model for generating long-form stories.
* **Action**:

  * Used **transformer causal language models**.
  * Tested varying context sizes and prompt strategies.
* **Result**: Generated coherent multi-paragraph stories; longer context improved narrative consistency.
* **Tags**: `Transformers` · `Causal LM` · `Text Generation` · `HuggingFace` · `Long-Context Modeling`

 ![Image](https://assets.imagineforest.com/blog/wp-content/uploads/2020/01/online-story-cube-generator.jpg)

---

### 🔮 Multi-Modal AI

#### 12. [AI Imagining Stories from Images](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Multi-modal/Ai_Imagining_Stories_from_Images.ipynb)

* **Situation**: Bridging computer vision with NLP enables storytelling from images.
* **Task**: Generate stories based on input images.
* **Action**:

  * Extracted image features via pretrained vision encoders.
  * Used HuggingFace causal language models to generate narratives.
* **Result**: Produced imaginative, contextually relevant stories from fantasy to real-world scenes.
* **Tags**: `Multi-Modal AI` · `Vision + Language` · `Transformers` · `HuggingFace` · `Image-to-Text`

 ![Image](https://static.vecteezy.com/system/resources/previews/035/898/060/large_2x/ai-generated-little-boy-reading-a-book-in-his-bedroom-fairy-tale-concept-a-child-s-imagination-being-fuelled-by-a-story-from-teacher-ai-generated-free-photo.jpg)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
