import tensorflow_hub as hub
import pandas as pd

def generate_text_feature(transcrpit_file_path):
    model_path = "/home/keyan/workspace/DepressionRec/models/universal-sentence-encoder-large_5"
    use_embed_large = hub.load(model_path)
    text_df = pd.read_csv(transcrpit_file_path, sep='\t').fillna('')
    sentences = []
    for t in text_df.itertuples():
        sentences.append(getattr(t, 'value'))
    text_feature = use_embed_large(sentences)
    return text_feature
