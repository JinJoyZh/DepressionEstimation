import os
import tensorflow_hub as hub
import pandas as pd

def analyze_text_feature(transcrpit_file_path):
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(root_dir, "weights/openface/bin", "universal-sentence-encoder-large_5")
    use_embed_large = hub.load(model_path)
    text_df = pd.read_csv(transcrpit_file_path)
    sentences = []
    for index, row in text_df.iterrows():
        sentences.append(row['value'])
    text_feature = use_embed_large(sentences)
    return text_feature

if __name__ == "__main__":
    transcrpit_file_path = '/home/zjy/workspace/tmp/interviewee_12345_1692723605/transcript.csv'
    analyze_text_feature(transcrpit_file_path)
