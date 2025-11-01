This subfolder contains the training pipelines for our BiLSTM model. It mainly consists of notebooks and a main predefined BiLSTM module for easy reference. Due to github restrictions, our saved models and clean data cannot be put onto github. In each pipeline, the references to our saved models will be denoted as `best_ckpts` which contains the trained BiLSTM model and `full_220` which contains our trained Word2Vec embeddings and Sentencepiece Processor

1) `biLSTM_pipeline.ipynb` : The main training pipeline for our base BiLSTM model, the final model is saved as a checkpoint 
2) `biLSTM_ablation.ipynb` : The ablation training pipeline for our BiLSTM encoder + HistGradientBoosy classifier head 
3) `biLSTM_reloader.ipynb` : This notebook was designed to reload both models for downstream evaluation testing
4) `biLSTM_training_eda.ipynb` : This notebook was used to analyse the training data and relevant plots included in our report 
5) `biLSTM.py` : This module serves as the central source of our BiLSTM module, the decision to modularise our model definition was intentional so as to support our 1 notebook per pipeline project design