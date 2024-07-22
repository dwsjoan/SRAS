import logging
import re
import threading
import sys
from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from gevent.pywsgi import WSGIServer, WSGIHandler, LoggingLogAdapter
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings('ignore')
import stslib
import queue
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from stslib import cfg, tool
from stslib.cfg import ROOT_DIR
from utils import *
import torch
from torch.backends import cudnn

class CustomRequestHandler(WSGIHandler):
    def log_request(self):
        pass

# 配置日志
# 禁用 Werkzeug 默认的日志处理器
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)
app = Flask(__name__, static_folder=os.path.join(ROOT_DIR, 'static'), static_url_path='/static',  template_folder=os.path.join(ROOT_DIR, 'templates'))
root_log = logging.getLogger()  # Flask的根日志记录器
root_log.handlers = []
root_log.setLevel(logging.WARNING)

# 配置日志
app.logger.setLevel(logging.WARNING)  # 设置日志级别为 INFO
# 创建 RotatingFileHandler 对象，设置写入的文件路径和大小限制
file_handler = RotatingFileHandler(os.path.join(ROOT_DIR, 'sts.log'), maxBytes=1024 * 1024, backupCount=5)
# 创建日志的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置文件处理器的级别和格式
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
# 将文件处理器添加到日志记录器中
app.logger.addHandler(file_handler)

funasr_lock = threading.RLock()


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


@app.route('/')
def index():
    sets = cfg.parse_ini()
    return render_template("index.html",
                           lang_code=cfg.lang_code,
                           language=cfg.LANG,
                           version=stslib.version_str,
                           p_num=sets.get('funasr_num_pipelines'),
                           root_dir=ROOT_DIR.replace('\\', '/'))

# 上传音频
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # 获取上传的文件
        audio_file = request.files['audio']
        noextname, ext = os.path.splitext(audio_file.filename)
        ext = ext.lower()
        wav_file = os.path.join(cfg.TMP_DIR, f'{noextname}.wav')
        if os.path.exists(wav_file) and os.path.getsize(wav_file) > 0:
            return jsonify({'code': 0, 'msg': cfg.transobj['lang1'], "data": os.path.basename(wav_file)})
        msg = ""
        # 如果是视频，先分离
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.mpeg', '.mp3', '.flac']:
            video_file = os.path.join(cfg.TMP_DIR, f'{noextname}{ext}')
            audio_file.save(video_file)
            params = [
                "-i",
                video_file,
            ]
            if ext not in ['.mp3', '.flac']:
                params.append('-vn')
            params.append(wav_file)
            rs = tool.runffmpeg(params)
            if rs != 'ok':
                return jsonify({"code": 1, "msg": rs})
            msg = "," + cfg.transobj['lang9']
        elif ext == '.wav':
            audio_file.save(wav_file)
        else:
            return jsonify({"code": 1, "msg": f"{cfg.transobj['lang3']} {ext}"})

        return jsonify({'code': 0, 'msg': cfg.transobj['lang1'] + msg, "data": os.path.basename(wav_file)})
    except Exception as e:
        app.logger.error(f'[upload]error: {e}')
        return jsonify({'code': 2, 'msg': cfg.transobj['lang2']})

def recognize(*, wav_name=None, asrmodel=None, language=None, data_type=None, wav_file=None, key=None, aisummary=None, qianfan_apikey, qianfan_secretkey,gemini_apikey, spkdi, device, web_l):
    try:
        if device == "cuda":
            if torch.cuda.is_available():
                if cudnn.is_available() and cudnn.is_acceptable(torch.tensor(1.).cuda()):
                    print('cuda和cudnn可用')
                    print('如果实际使用仍提示cuda相关错误，请尝试升级显卡驱动')
                else:
                    cfg.progressbar[key] = 1
                    cfg.progressresult[key] = 'cuda可用但cudnn不可用，cuda11.x请安装cudnn8,cuda12.x请安装cudnn9'
                    return
            else:
                cfg.progressbar[key] = 1
                cfg.progressresult[key] = "当前计算机CUDA不可用，torch.cuda.is_available()返回False"
                return
        if aisummary != "no":
            if aisummary=="ernie_speed":
                response=qianfan_access_token_test(qianfan_apikey, qianfan_secretkey)
                print(response)
                if not response.get("access_token"):
                    error_response = "百度千帆API错误" if web_l == "zh" else "Baidu Qianfan API ERROR"
                    for k,v in response.items():
                        error_response = error_response + "\n" + k + "：" + str(v)
                    cfg.progressbar[key] = 1
                    cfg.progressresult[key] = error_response
                    return
            if aisummary=="gemini_flash":
                response = gemini_access_token_test(gemini_apikey)
                print(response)
                if response.get("error"):
                    error_response = "GeminiAPI错误" if web_l == "zh" else "Gemini API ERROR"
                    for k, v in response.get("error").items():
                        error_response = error_response + "\n" + k + "：" + str(v)
                    cfg.progressbar[key] = 1
                    cfg.progressresult[key] = error_response
                    return
        if "funasr" in asrmodel:
            if language != "zhs":
                cfg.progressbar[key] = 1
                cfg.progressresult[key] = "Please try a different model"
                return
            elif asrmodel == "funasr":
                cfg.specificprogress[key] = "等待funasr模型中..." if web_l == "zh" else "Waiting for the funasr model in..."
                funasr_lock.acquire()
                res=funasr_singlefile(wav_file, device, key, web_l)
                funasr_lock.release()
            else:
                if not cfg.pipelien_exist:
                    cfg.pipelien_exist = True
                    funasr_pipeline_create(device, key, web_l)
                else:
                    cfg.specificprogress[key] ="等待funasr pipeline中..." if web_l == "zh" else "Waiting for funasr pipeline in..."
                res = funasr_mutifile(wav_file, device, key, web_l)
            if data_type == "text":
                if spkdi != "no":
                    cfg.specificprogress[key] = "识别结果转化text格式中..." if web_l == "zh" else "Recognize results in text format..."
                    txt_spk_mapping, result_texts = funasr_speaker_map(res, key, language)
                    result_texts = "\n".join(result_texts)
                else:
                    cfg.specificprogress[key] = "识别结果转化text格式中..." if web_l == "zh" else "Recognize results in text format..."
                    result_texts = funasr_nospk(res, key)
                    result_texts = "\n".join(result_texts)
                if aisummary != "no":
                    cfg.specificprogress[key] = "AI总结中..." if web_l == "zh" else "AI summarizes in..."
                    if aisummary=="ernie_speed":
                        aisummary_texts = ernie_speed_128k_summary(result_texts, qianfan_apikey, qianfan_secretkey, language)
                    else:
                        aisummary_texts = gemini_flash_summary(result_texts, gemini_apikey, language)
                    head = "总结如下：\n\n" if "zh" in language else "Summarized below:\n\n"
                    mid = "对话原文如下：\n\n" if "zh" in language else "The audio text is below:\n\n"
                    result_texts = head + aisummary_texts + "\n\n--------------------------------------------------\n\n"+ mid + result_texts
            elif data_type == "json":
                cfg.specificprogress[key] = "识别结果转化json格式中..." if web_l == "zh" else "Recognize results in json format..."
                result_texts = funasr_json(res, key)
            else:
                cfg.specificprogress[key] = "识别结果转化srt格式中..." if web_l == "zh" else "Recognize results in srt format..."
                result_texts = funasr_srt(res, key)
                result_texts = "\n".join(result_texts)
        else:
            nemo_process = nemo_detect(asrmodel, data_type, spkdi, wav_file, key, device)
            segments, info = whisper(asrmodel, key, wav_file, language, device, web_l)
            if data_type == "text":
                head_punc=""
                if spkdi != "no":
                    cfg.specificprogress[key] = "NeMo模型运行中..." if web_l == "zh" else "NeMo model run in..."
                    nemo_process.communicate()
                    cfg.specificprogress[key] = "NeMo结果处理中..." if web_l == "zh" else "NeMo results processing in..."
                    speaker_ts = nemo(key)
                    cfg.specificprogress[key] = "whisper结果处理中..." if web_l == "zh" else "whisper results in process..."
                    text_speaker_map, result_texts, head_punc = whisper_speaker_map(segments, speaker_ts, language, key, spkdi)
                    result_texts = "\n".join(result_texts)
                else:
                    cfg.specificprogress[key] = "识别结果转化text格式中..." if web_l == "zh" else "Recognize results in text format..."
                    result_texts = whisper_nospk(segments, key, language)
                    result_texts = "\n".join(result_texts)
                if aisummary != "no":
                    cfg.specificprogress[key] = "AI总结中..." if web_l == "zh" else "AI summarizes in..."
                    if aisummary == "ernie_speed":
                        aisummary_texts = ernie_speed_128k_summary(result_texts, qianfan_apikey, qianfan_secretkey, language)
                    else:
                        aisummary_texts = gemini_flash_summary(result_texts, gemini_apikey, language)
                    head = "总结如下：\n\n" if "zh" in language else "Summarized below:\n\n"
                    mid = "对话原文如下：\n\n" if "zh" in language else "The audio text is below:\n\n"
                    result_texts = head_punc + head + aisummary_texts + "\n\n--------------------------------------------------\n\n" + mid + result_texts
                else:
                    result_texts = head_punc + result_texts
            elif data_type == "json":
                cfg.specificprogress[key] = "识别结果转化json格式中..." if web_l == "zh" else "Recognize results in json format..."
                result_texts = whisper_json(segments, key, language)
            else:
                cfg.specificprogress[key] = "识别结果转化srt格式中..." if web_l == "zh" else "Recognize results in srt format..."
                result_texts = whisper_srt(segments, key, language)
                result_texts = "\n".join(result_texts)
        cfg.specificprogress[key] = "完成" if web_l == "zh" else "Done."
        cfg.progressbar[key] = 1
        cfg.progressresult[key] = result_texts
    except Exception as e:
        cfg.progressresult[key]=str(e)
        print(str(e))

# name=文件名字，filename=文件绝对路径
# wav_name:tmp下的wav文件
@app.route('/process', methods=['GET', 'POST'])
def process():
    # 原始字符串
    wav_name = request.form.get("wav_name").strip()
    asrmodel = request.form.get("asrmodel")
    device = request.form.get("device")
    language = request.form.get("language")
    qianfan_apikey = request.form.get("qianfan_apikey")
    qianfan_secretkey = request.form.get("qianfan_SecretKey")
    gemini_apikey=request.form.get("gemini_apikey")
    data_type = request.form.get("data_type")
    aisummary = request.form.get("aisummary")
    spkdi = request.form.get("spkdi")
    file_num = int(request.form.get("file_num"))
    web_l = request.form.get("web_language")
    wav_file = os.path.join(cfg.TMP_DIR, wav_name)
    key = f'{wav_name}{asrmodel}{language}{data_type}'
    cfg.progressresult[key] = None
    cfg.progressbar[key] = 0
    if web_l == "zh":
        cfg.specificprogress[key] = '等待处理'
    else:
        cfg.specificprogress[key] ="Awaiting processing"
    if not os.path.exists(wav_file):
        return jsonify({"code": 1, "msg": f"{wav_file} {cfg.langlist['lang5']}"})
    if "funasr" not in asrmodel:
        if not os.path.exists(os.path.join(cfg.MODEL_DIR, f'models--Systran--faster-whisper-{asrmodel}/snapshots/')):
            return jsonify({"code": 1, "msg": f"{asrmodel} {cfg.transobj['lang4']}"})
    threading.Thread(target=recognize, kwargs={"wav_name":wav_name, "asrmodel":asrmodel, "language":language, "data_type":data_type, "wav_file":wav_file, "key":key, "aisummary":aisummary, "qianfan_apikey":qianfan_apikey, "qianfan_secretkey":qianfan_secretkey, "gemini_apikey":gemini_apikey, "spkdi":spkdi, "device":device, "web_l":web_l}).start()
    return jsonify({"code":0, "msg":"ing"})

# 获取进度及完成后的结果
@app.route('/progressbar', methods=['GET', 'POST'])
def progressbar():
    wav_name = request.form.get("wav_name").strip()
    asrmodel = request.form.get("asrmodel")
    language = request.form.get("language")
    data_type = request.form.get("data_type")
    key = f'{wav_name}{asrmodel}{language}{data_type}'
    progressbar = cfg.progressbar[key]
    specificprogress = cfg.specificprogress[key]
    if progressbar>=1:
        return jsonify({"code":0, "data":progressbar, "msg":"ok", "result":cfg.progressresult[key]})
    return jsonify({"code":0, "specificprogress":specificprogress, "msg":"ok"})

@app.route('/checkupdate', methods=['GET', 'POST'])
def checkupdate():
    return jsonify({'code': 0, "msg": cfg.updatetips})

if __name__ == '__main__':
    http_server = None
    try:
        threading.Thread(target=tool.checkupdate).start()
        try:
            host = cfg.web_address.split(':')
            http_server = WSGIServer((host[0], int(host[1])), app, handler_class=CustomRequestHandler)
            threading.Thread(target=tool.openweb, args=(cfg.web_address,)).start()
            http_server.serve_forever()
        finally:
            if http_server:
                http_server.stop()
    except Exception as e:
        if http_server:
            http_server.stop()
        print("error:" + str(e))
        app.logger.error(f"[app]start error:{str(e)}")
