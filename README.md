# 🧠 AI Projects: NLP & Computer Vision

This repository is a curated collection of **end-to-end AI projects** spanning **Computer Vision**, **Natural Language Processing (NLP)**, and **Multi-Modal AI**. Each project is implemented in a **self-contained Jupyter Notebook** (or script) with explanations, code, and results.

The repo is structured to demonstrate both **breadth of exploration** and **depth of implementation**, making it suitable as both a **learning hub** and a **portfolio showcase**.

🔗 Useful Links:

* 📂 [GitHub Repo](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision)
* 🤖 [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* 🖼️ [Keras Applications](https://keras.io/api/applications/)
* 📊 [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
* 📸 [Flickr30k Dataset](https://shannon.cs.illinois.edu/DenotationGraph/)

---

## 📂 Projects

### 🖼️ Computer Vision

1. [**Face Mask Detection with VGG16**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Face_Mask_Detection_with_VGG16.ipynb)

   * **Situation**: Needed during COVID-19 to monitor mask compliance.
   * **Task**: Detect whether a person is wearing a mask.
   * **Action**: Transfer learning with **VGG16** + image augmentation.
   * **Result**: Achieved high-accuracy mask detection suitable for real-time monitoring.

2. [**Facial Emotion Recognition with VGG16**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Facial_Emotion_Recognition_with_VGG16.ipynb)

   * **Situation**: Emotion recognition is key for healthcare & user experience.
   * **Task**: Classify facial expressions into emotion categories.
   * **Action**: Fine-tuned **VGG16** on FER datasets with categorical encoding.
   * **Result**: Model correctly classifies multiple emotions (happy, sad, angry, etc.).

3. [**Fashion MNIST Classification with CNNs**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Fashion_MNIST_with_CNN%28s%29.ipynb)

   * **Situation**: Benchmark dataset for image classification.
   * **Task**: Train CNNs to classify clothes into 10 categories.
   * **Action**: Built multiple CNN architectures, trained & evaluated on Fashion-MNIST.
   * **Result**: Achieved >90% accuracy on test set.

4. [**Green Screening with OpenCV**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Green_Screening_Images_and_Videos_with_OpenCV.ipynb)

   * **Situation**: Green screening (chroma keying) widely used in media production.
   * **Task**: Replace backgrounds in images & videos.
   * **Action**: Implemented background segmentation & replacement using **OpenCV**.
   * **Result**: Real-time replacement of green background with chosen scenes.

5. [**Image Deblurring with VGG16 + DCGAN**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Image_Deblurring_with_VGG16.ipynb)

   * **Situation**: Blurred images reduce clarity in photos, surveillance, and medical imaging.
   * **Task**: Restore sharpness of blurred images.
   * **Action**: Built a **DCGAN-based model** enhanced with **VGG16 feature loss**.
   * **Result**: Restored sharper and more visually appealing images.

6. [**Image Captioning with Flickr30k**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Image_Captioning_with_Flickr30k.ipynb)

   * **Situation**: Image captioning supports accessibility & media search.
   * **Task**: Train model to describe images in words.
   * **Action**: Combined **VGG16 feature extraction** with **LSTMs** for sequence generation.
   * **Result**: Generated coherent captions (e.g., “A dog playing with a child”).

---

### 📝 Natural Language Processing (NLP)

7. [**Tweets Sentiment Analysis with 3 Neural Networks**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Tweets_Sentiment_Analysis_with_3_Neural_Network.ipynb)

   * **Situation**: Tweets reflect diverse public sentiment.
   * **Task**: Build sentiment classifiers with neural networks.
   * **Action**: Designed and trained 3 **DNN architectures** with preprocessing pipeline.
   * **Result**: Successfully classified tweets into positive/negative/neutral categories.

8. [**GenZ Tweets Data Pipeline for Sentiment Analysis**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/GenZ_Tweets_Data_Pipeline_for_Sentiment_Analysis.ipynb)

   * **Situation**: Real-world data is noisy and unstructured.
   * **Task**: Build robust preprocessing pipeline for tweets.
   * **Action**: Used **NLTK + SpaCy** for tokenization, lemmatization, stopword removal, regex cleaning, and emoji normalization.
   * **Result**: Produced clean, structured data improving downstream ML performance.

9. [**Next Word Prediction with Bi-Directional LSTM**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Next_Word_Prediction_with_Bidirectional_LSTM.ipynb)

   * **Situation**: Predicting the next word is core to autocomplete & search.
   * **Task**: Train a language model to predict next words.
   * **Action**: Implemented **Bi-LSTM sequence model** on text data with embeddings.
   * **Result**: Generated context-aware predictions for text completion.

10. [**Prompt-to-Synopsis Generator (Fine-Tuning)**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Prompt_to_Synopsis_Generator_%28Fine-Tuning%29.ipynb)

* **Situation**: Creative industries need AI that can expand prompts into stories.
* **Task**: Fine-tune a transformer model to generate synopses from prompts.
* **Action**: Used **HuggingFace Transformers** to fine-tune, evaluate, and test prompts.
* **Result**: Produced structured multi-sentence synopses from one-line prompts.

11. [**AI Long-Form Story Generator with Varied Context**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Ai_Long_form_Story_Generator_with_Varied_Context.ipynb)

* **Situation**: Longer context windows improve story coherence.
* **Task**: Generate long narratives conditioned on different context lengths.
* **Action**: Built story generator using **transformer-based causal models** with varied prompt designs.
* **Result**: Generated coherent long-form stories, balancing creativity & coherence.

---

### 🔮 Multi-Modal AI

12. [**AI Imagining Stories from Images**](https://github.com/sanskarGupta551/ai-projects-nlp-computer-vision/blob/main/Ai_Imagining_Stories_from_Images.ipynb)

* **Situation**: Images often hide stories that can be narrated.
* **Task**: Build AI that generates stories based on images.
* **Action**: Combined **vision encoders** with **causal language models** to generate stories from input images.
* **Result**: Produced creative and contextually relevant narratives for diverse images (fantasy scenes, professional settings, etc.).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
