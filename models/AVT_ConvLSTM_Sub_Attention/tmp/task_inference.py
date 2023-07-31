import time

from autolab_core import YamlConfig

# local functions

from models.AVT_ConvLSTM_Attention.utils import *


def main(data_loader, visual_net, audio_net, text_net, evaluator, config, device):
    if not config['CRITERION']['USE_SOFT_LABEL']:
        assert config['EVALUATOR']['CLASSES_RESOLUTION'] == config['EVALUATOR']['N_CLASSES'], \
            "Argument --config['EVALUATOR']['CLASSES_RESOLUTION'] should be the same as --config['EVALUATOR']['N_CLASSES'] when soft label is not used!"
    mode_start_time = time.time()
    #init pred list
    phq_score_pred = []
    phq_subscores_pred = []
    phq_binary_pred = []
    #eval mode for models
    visual_net.eval()
    audio_net.eval()
    # text_net.eval()
    evaluator.eval()
    torch.set_grad_enabled(False)

    # TODO: extract features with multi-model ...
    # combine all models into a function
    def model_processing(input):
        # get facial visual feature with Deep Visual Net'
        # input shape for visual_net must be (B, C, F, T) = (batch_size, channels, features, time series)
        B, T, F, C = input['visual'].shape
        visual_input = input['visual'].permute(0, 3, 2, 1).contiguous()
        visual_features = visual_net(visual_input.to(device))  # output dim: [B, visual net output dim]

        # get audio feature with Deep Audio Net'
        # input shape for audio_net must be (B, F, T) = (batch_size, features, time series)
        B, F, T = input['audio'].shape
        audio_input = input['audio'].view(B, F, T)
        audio_features = audio_net(audio_input.to(device))  # output dim: [B, audio net output dim]

        # get Text features with Deep Text Net'
        # input shape for text_net must be (B, F, T) = (batch_size, features, time series))
        B, T, F = input['text'].shape
        text_input = input['text'].permute(0, 2, 1).contiguous()
        text_features = text_net(text_input.to(device))  # output dim: [B, text net output dim]

        # ---------------------- Start evaluating with sub-attentional feature fusion ----------------------
        # combine all features into shape: B, C=1, num_modal, audio net output dim
        all_features = torch.stack([visual_features, audio_features, text_features], dim=1).unsqueeze(dim=1)
        probs = evaluator(all_features)
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

    probs = model_processing(input=data)
    # predict the final score
    pred_score = compute_score_with_args(probs, config['EVALUATOR'], device)
    phq_score_pred.extend([pred_score[i].item() for i in range(batch_size)])  # 1D list
    phq_binary_pred.extend([1 if pred_score[i].item() >= config['PHQ_THRESHOLD'] else 0 for i in range(batch_size)])


def run_task():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config_file = '../config/config_inference.yaml'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = '2,3'
    # set up GPU
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    # load config file into dict() format
    config = YamlConfig(config_file)
    # create the output folder (name of experiment) for storing model result such as logger information
    if not os.path.exists(config['OUTPUT_DIR']):
        os.mkdir(config['OUTPUT_DIR'])
    # print configuration
    print(config.file_contents)
    config.save(os.path.join(config['OUTPUT_DIR'], config['SAVE_CONFIG_NAME']))
    print('=' * 40)
    # initialize random seed for torch and numpy
    init_seed(config['MANUAL_SEED'])
    # get logger os.path.join(config['OUTPUT_DIR'], f'{config['TYPE']}_{config['LOG_TITLE']}.log')
    file_name = os.path.join(config['OUTPUT_DIR'], '{}.log'.format(config['TYPE']))
    # base_logger = get_logger(file_name, config['LOG_TITLE'])
    # get dataloaders
    dataloaders = get_dataloaders(config['DATA'])
    # get models
    ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
    model_type = config['TYPE']
    visual_net, audio_net, text_net, evaluator = get_models(config['MODEL'], gpu, device, model_type, ckpt_path)
    main(dataloaders, visual_net, audio_net, text_net, evaluator, config['MODEL'], device)
