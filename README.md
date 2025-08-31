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

#### 1. [Face Mask Detection with VGG16](Computer-Vision/Face_Mask_Detection_with_VGG16.ipynb)

* **Situation**: COVID-19 demanded scalable mask compliance monitoring.
* **Task**: Detect whether a person is wearing a mask.
* **Action**: Fine-tuned VGG16 with transfer learning, applied augmentation, trained on \~12K Kaggle dataset.
* **Result**: Achieved **97% validation accuracy**.
* **Tags**: `TensorFlow` · `Keras` · `Transfer Learning` · `CNN` · `Image Classification`
  ![Image](https://www.intertecsystems.com/wp-content/uploads/2020/05/face-mask-detection-software-e1591538656411.png)

---

#### 2. [Facial Emotion Recognition with VGG16](Computer-Vision/Facial_Emotion_Recognition_with_VGG16.ipynb)

* **Situation**: Emotion recognition is vital for healthcare, customer analytics, and HCI.
* **Task**: Classify facial emotions into discrete categories.
* **Action**: Adapted VGG16 (Imagenet base), trained on FER2013 dataset with Haar cascade preprocessing.
* **Result**: Achieved **72% accuracy**, significantly above classical baselines.
* **Tags**: `Keras` · `VGG16` · `FER-2013` · `Emotion Recognition` · `Transfer Learning`
  ![Image](https://i.pinimg.com/originals/b0/bb/1d/b0bb1d0b86bdca1de8ead928064d09d8.png)

---

#### 3. [Fashion MNIST Classification with CNNs](Computer-Vision/Fashion_MNIST_with_CNN%28s%29.ipynb)

* **Situation**: Benchmark dataset for image classification.
* **Task**: Classify 28x28 grayscale clothing images into 10 categories.
* **Action**: Built multiple CNNs of varying depth, evaluated performance.
* **Result**: Best CNN achieved **92% accuracy** on test set.
* **Tags**: `TensorFlow` · `CNN` · `Fashion-MNIST` · `Model Comparison`
  ![Image](https://thiagolcmelo.github.io/assets/img/fashion-mnist.png)

---

#### 4. [Green Screening with OpenCV](Computer-Vision/Green_Screening_Images_and_Videos_with_OpenCV.ipynb)

* **Situation**: Green screening (chroma keying) is used in film, media, and AR.
* **Task**: Replace green backgrounds in media with new scenes.
* **Action**: Applied OpenCV masking on green pixels; tested on both images and videos.
* **Result**: Delivered smooth real-time background replacement.
* **Tags**: `OpenCV` · `Chroma Keying` · `Real-Time Processing` · `Computer Vision`
  ![Image](https://i.ytimg.com/vi/JfyzwWgZT4M/maxresdefault.jpg)

---

#### 5. [Image Deblurring with VGG16 + DCGAN](Computer-Vision/Image_Deblurring_with_VGG16.ipynb)

* **Situation**: Blur reduces image clarity in surveillance and healthcare.
* **Task**: Restore sharpness of blurred images.
* **Action**: Built mixed-blur dataset; trained DCGAN with VGG16 perceptual loss.
* **Result**: Improved SSIM by **+18% over baseline interpolation**.
* **Tags**: `GAN` · `DCGAN` · `VGG16` · `Image Restoration` · `Perceptual Loss`
  ![Image](https://th.bing.com/th/id/R.1c95a17451c44bb937d2275bc6ae14d9?rik=FClK479meeWRPA\&riu=http%3a%2f%2fwww.ece.northwestern.edu%2flocal-apps%2fmatlabhelp%2ftoolbox%2fimages%2fdeblu10a.gif\&ehk=VTBAkwI%2bK%2b2lqG5JtdUoByh24GOJfmcoo8RGX8xAyiY%3d\&risl=\&pid=ImgRaw\&r=0)

---

#### 6. [Image Captioning with Flickr30k](Computer-Vision/Image_Captioning_with_Flickr30k.ipynb)

* **Situation**: Automatic captioning helps accessibility and content indexing.
* **Task**: Train a model to generate captions for images.
* **Action**: Built encoder-decoder model (VGG16 + LSTM) trained on Flickr30k dataset.
* **Result**: Generated fluent captions, BLEU-4 ≈ **0.41**.
* **Tags**: `VGG16` · `LSTM` · `Seq2Seq` · `Encoder-Decoder` · `Image Captioning`
  ![Image](https://petapixel.com/assets/uploads/2016/09/Caption1-800x450.jpg)

---

### 📝 Natural Language Processing (NLP)

#### 7. [Tweets Sentiment Analysis with 3 Neural Networks](NLP/Tweets_Sentiment_Analysis_with_3_Neural_Network.ipynb)

* **Situation**: Sentiment insights from Twitter are valuable for marketing and politics.
* **Task**: Build and compare multiple deep learning models for sentiment classification.
* **Action**: Preprocessed text with regex, embeddings; trained DNN, CNN, and RNN.
* **Result**: CNN achieved **88% accuracy**, outperforming other architectures.
* **Tags**: `NLP` · `Sentiment Analysis` · `DNN` · `CNN` · `RNN` · `Embeddings`
  ![Image](https://www.altexsoft.com/media/2018/09/sentiment_analysis.jpg)

---

#### 8. [GenZ Tweets Data Pipeline for Sentiment Analysis](NLP/GenZ_Tweets_Data_Pipeline_for_Sentiment_Analysis.ipynb)

* **Situation**: Social media data is noisy with slang, emojis, and hashtags.
* **Task**: Build preprocessing pipeline to clean GenZ tweets.
* **Action**: Regex cleaning, tokenization, lemmatization, emoji & URL normalization.
* **Result**: Improved downstream sentiment model accuracy by \~10%.
* **Tags**: `NLP` · `Data Pipeline` · `Regex` · `NLTK` · `SpaCy`
  ![Image](https://i.pinimg.com/originals/e1/d0/33/e1d0330eb3bfd698f6c332e0d61f6d11.png)

---

#### 9. [Next Word Prediction with Bi-LSTM](NLP/Next_Word_Prediction_with_Bidirectional_LSTM.ipynb)

* **Situation**: Next word prediction is a core task for autocomplete and search engines.
* **Task**: Train Bi-LSTM model for next-word prediction.
* **Action**: Preprocessed text corpus into sequences, trained Bi-LSTM with embeddings.
* **Result**: Reduced perplexity to **\~35**; generated contextually accurate predictions.
* **Tags**: `Bi-LSTM` · `Language Modeling` · `Sequence Prediction` · `Text Generation`
  ![Image](https://i.pinimg.com/originals/3f/b4/2e/3fb42e1f042063e0423496529978dd8e.jpg)

---

#### 10. [Prompt-to-Synopsis Generator (Fine-Tuning)](NLP/Prompt_to_Synopsis_Generator_%28Fine-Tuning%29.ipynb)

* **Situation**: Entertainment requires auto-expansion of ideas into story outlines.
* **Task**: Fine-tune a transformer to generate synopses from prompts.
* **Action**: Fine-tuned GPT-2 using HuggingFace; applied causal LM loss and LR scheduling.
* **Result**: Generated coherent multi-sentence synopses.
* **Tags**: `Transformers` · `GPT-2` · `Fine-Tuning` · `Text Generation`
  ![Image](https://th.bing.com/th/id/R.62fed4c6ba6af08871ed40c89f4d0a44?rik=bHeTpfghWffTSQ&riu=http%3a%2f%2fsusancushman.com%2fwp-content%2fuploads%2f2012%2f10%2f100_12691.jpg&ehk=u%2fiowcjKxXZcabvS21Q2ExLG0g6YsIK6vJcgkcE7Xxc%3d&risl=&pid=ImgRaw&r=0)

---

#### 11. [AI Long-Form Story Generator with Varied Context](NLP/Ai_Long_form_Story_Generator_with_Varied_Context.ipynb)

* **Situation**: Long-context transformers improve narrative flow.
* **Task**: Generate long-form stories using varying input context.
* **Action**: Built transformer causal LM, tested multiple prompt sizes.
* **Result**: Produced coherent multi-paragraph stories with consistent characters.
* **Tags**: `Transformers` · `Causal LM` · `Long-Context Modeling` · `Text Generation`
  ![Image](https://miro.medium.com/v2/resize\:fit:1400/1*lCOIr6tlsKd8N247Ec9MBg.png)

---

### 🔮 Multi-Modal AI

#### 12. [AI Imagining Stories from Images](Multi-modal/Ai_Imagining_Stories_from_Images.ipynb)

* **Situation**: Images often imply hidden narratives.
* **Task**: Generate creative stories from input images.
* **Action**: Used HuggingFace pipelines (vision encoders + causal LM).
* **Result**: Produced imaginative narratives across diverse images.
* **Tags**: `Multi-Modal AI` · `Vision+Language` · `Transformers` · `Story Generation`
  ![Image](https://medium.com/images/ai-image-to-story-example.jpg)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
