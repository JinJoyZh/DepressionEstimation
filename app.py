import os

from flask import Flask, jsonify, request

from diagnostic import MultimodalDiagnostic


app = Flask(__name__)


@app.route('/')
def hello():
    user_data_dir = "/app/interviewee_12345_1692723605"
    serivce = MultimodalDiagnostic(user_data_dir)
    print('start to get video feature')

    # 以下出现四次语音片段，每次都需要 1.提取video_features 2. 记录文字内容
    # case 1
    serivce.generate_video_features("vdieo_1692758150.wmv")
    # serivce.transcript("1692758150", "1692758250", "aaaaa")
    # case 2
    serivce.generate_video_features("video_1692757670.wmv")
    # serivce.transcript("1692757670", "1692757770", "bbbbb")
    # case 3
    serivce.generate_video_features("video_1692757850.wmv")
    # serivce.transcript("1692757850, "1692757950", "bbbbb")
    # case 4
    serivce.generate_video_features("video_1693300015.wmv")
    # serivce.transcript("1692757850, "1692757950", "bbbbb")

    # 生成PHQ
    video_frame_rate = 30
    phq_score_pred, phq_binary_pred = serivce.generate_phq(30)
    return jsonify({"phqScore": phq_score_pred,
                    "phqBinary": phq_binary_pred})


@app.route('/predict', methods=['POST'])
def multi_model_predict():
    content = request.json
    source = os.getenv("SOURCE_FOLDER", "/app/source")
    root_folder = os.path.join(source,content['rootFolder'])
    print(f"save to folder {root_folder} \n")
    serivce = MultimodalDiagnostic(root_folder)
    conversations = content['conversations']
    for c in conversations:
        video = c['video']
        content = c['content']
        start = c['start']
        end = c['end']
        print(f"receive conversation {c} \n")
        serivce.generate_video_features(f"{video}.wmv")
        serivce.transcript(str(start), str(end), content)
    phq_score_pred, phq_binary_pred = serivce.generate_phq(30)
    return jsonify({"phqScore": 0,
                    "phqBinary": 1.0})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5050)