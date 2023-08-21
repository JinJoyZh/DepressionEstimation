import os
import tensorflow_hub as hub
import pandas as pd

def analyze_text_feature(transcrpit_file_path):
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(root_dir, "bin", "universal-sentence-encoder-large_5")
    use_embed_large = hub.load(model_path)
    text_df = pd.read_csv(transcrpit_file_path, sep='\t').fillna('')
    sentences = []
    for t in text_df.itertuples():
        sentences.append(getattr(t, 'value'))
    text_feature = use_embed_large(sentences)
    return text_feature

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(root_dir, "bin", "universal-sentence-encoder-large_5")
    print(model_path)
