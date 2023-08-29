import os
import time
import torch
import sys

sys.path.append("..") 
from utils import get_models


def model_processing(input, config, args):
    start_time = time.time()
    # get logger os.path.join(config['OUTPUT_DIR'], f'{config['TYPE']}_{config['LOG_TITLE']}.log')
    file_name = os.path.join(config['OUTPUT_DIR'], '{}.log'.format(config['TYPE']))
    # base_logger = get_logger(file_name, config['LOG_TITLE'])
    # get models
    ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
    model_type = config['TYPE']
    visual_net, audio_net, text_net, evaluator = get_models(config['MODEL'], args, model_type, ckpt_path)
    visual_net.eval()
    audio_net.eval()
    text_net.eval()
    evaluator.eval()
    torch.set_grad_enabled(False)
    # get facial visual feature with Deep Visual Net'
    # input shape for visual_net must be (B, C, F, T) = (batch_size, channels, features, time series)
    B, T, F, C = input['visual'].shape
    visual_input = input['visual'].permute(0, 3, 2, 1).contiguous()
    visual_features = visual_net(visual_input.to(args.device))  # output dim: [B, visual net output dim]

    # get audio feature with Deep Audio Net'
    # input shape for audio_net must be (B, F, T) = (batch_size, features, time series)
    B, F, T = input['audio'].shape
    audio_input = input['audio'].view(B, F, T)
    audio_features = audio_net(audio_input.to(args.device))  # output dim: [B, audio net output dim]

    # get Text features with Deep Text Net'
    # input shape for text_net must be (B, F, T) = (batch_size, features, time series))
    B, T, F = input['text'].shape
    text_input = input['text'].permute(0, 2, 1).contiguous()
    text_features = text_net(text_input.to(args.device))  # output dim: [B, text net output dim]

    # ---------------------- Start evaluating with sub-attentional feature fusion ----------------------
    # combine all features into shape: B, C=1, num_modal, audio net output dim
    all_features = torch.stack([visual_features, audio_features, text_features], dim=1).unsqueeze(dim=1)
    probs = evaluator(all_features)
    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))
    """ 
    Arguments:
        'features' should have size (batch_size, channels(=1), num_modal, feature_dim of each branch)
    Output:
        if PREDICT_TYPE == phq-subscores:
            'probs' is a list of torch matrices
            len(probs) == number of subscores == 8
            probs[0].size() == (batch size, class resolution)
        else:
            'probs' a torch matrices with shape: (batch size, class resolution)
    """
    return probs
    



