import json
import wget
from omegaconf import OmegaConf
import os
from stslib import cfg
import opencc
import torch
from funasr import AutoModel
from pydub import AudioSegment
from faster_whisper import WhisperModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from stslib import cfg, tool
import queue
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import subprocess
import shutil
import sys

punct_model_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]
prompt_list={
        'zhs': "以下是普通话的句子，这是一段语音记录。",
        'zht': "以下係一段語音記錄。",
        'en': "The following is a transcript of the voice.",
        'fr': "Ce qui suit est une transcription de la voix.",
        'de': "Das Folgende ist eine Abschrift der Stimme.",
        'ja': "以下は、その声の書き起こしです。",
        'ko': "다음은 그 목소리의 녹취록이다.",
        'ru': "Ниже приводится расшифровка голоса.",
        'es': "La siguiente es una transcripción de la voz.",
        'th': "ต่อไปนี้เป็นการถอดเสียงของเสียง",
        'it': "Quella che segue è una trascrizione della voce.",
        'pt': "Segue-se uma transcrição da voz.",
        'vi': "Sau đây là bảng điểm của giọng nói.",
        'ar': "فيما يلي نص الصوت.",
        'tr': "Aşağıda sesin bir dökümü yer almaktadır.",
        'hu': "Az alábbiakban a hang átirata található.",
    }

pipeline_queue = queue.Queue()
def funasr_pipeline_create(device, key, web_l):
    global pipeline_queue
    sets = cfg.parse_ini()
    for _ in range(sets.get('funasr_num_pipelines')):
        cfg.specificprogress[key] =f"第{_+1}条funasr模型pipeline创建中..." if web_l == "zh" else f"Article {_+1} funasr model pipeline creation in progress..."
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='models/paraformer-zh',
            vad_model='models/fsmn-vad',
            punc_model='models/ct-punc-c',
            spk_model="models/cam++",
            device=device
        )
        pipeline_queue.put(inference_pipeline)
    cfg.specificprogress[key] = f"全部{sets.get('funasr_num_pipelines')}条funasr模型pipeline创建完毕" if web_l == "zh" else f"All {sets.get('funasr_num_pipelines')} funasr model pipelines created"

def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"  # Can be meeting, telephonic, or general based on domain type of the audio file
    CONFIG_LOCAL_DIRECTORY = "nemo_msdd_configs"
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    MODEL_CONFIG_PATH = os.path.join(CONFIG_LOCAL_DIRECTORY, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG_PATH):
        os.makedirs(CONFIG_LOCAL_DIRECTORY, exist_ok=True)
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        MODEL_CONFIG_PATH = wget.download(CONFIG_URL, MODEL_CONFIG_PATH)

    config = OmegaConf.load(MODEL_CONFIG_PATH)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"
    config.num_workers = 0
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config


def nemo_detect(asrmodel, data_type, spkdi, wav_file, key, device):
    if data_type != "text":
        return 0
    elif spkdi == "no":
        return 0
    elif "funasr" not in asrmodel:
        return subprocess.Popen(
                    [sys.executable, "nemo_process.py", "-a", wav_file, "-k", key, "--device", device],
                )
    else:
        return 0

def nemo(key):
    speaker_ts = []
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, key)
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
    shutil.rmtree(temp_path)
    return speaker_ts

def whisper(model,key,wav_file,language,device,web_l):
    sets = cfg.parse_ini()
    cfg.specificprogress[key] = "whisper模型加载中..." if web_l == "zh" else "whisper model loading..."
    modelobj = WhisperModel(model,
                            device=device,
                            compute_type=sets.get('cuda_com_type'),
                            download_root=cfg.ROOT_DIR + "/models", local_files_only=True)
    cfg.specificprogress[key] = "whisper模型运行中..." if web_l == "zh" else "whisper model is running in..."
    segments, info = modelobj.transcribe(wav_file,
                                         beam_size=sets.get('beam_size'),
                                         best_of=sets.get('best_of'),
                                         temperature=0 if sets.get('temperature') == 0 else [0.0, 0.2, 0.4, 0.6, 0.8,
                                                                                             1.0],
                                         condition_on_previous_text=sets.get('condition_on_previous_text'),
                                         vad_filter=sets.get('vad'),
                                         vad_parameters=dict(min_silence_duration_ms=300),
                                         language="zh" if "zh" in language else language,
                                         initial_prompt=prompt_list[language]
                                         )
    del modelobj
    torch.cuda.empty_cache()
    return segments,info

def funasr_singlefile(wav_file,device,key,web_l):
    if not cfg.funasr_exist:
        cfg.funasr_exist = True
        cfg.specificprogress[key] = "funaser模型加载中..." if web_l == "zh" else "funaser model loading..."
        cfg.funasr_model = AutoModel(model="models/paraformer-zh",
                                     vad_model="models/fsmn-vad",
                                     punc_model="models/ct-punc-c",
                                     spk_model="models/cam++",
                                     device=device,
                                     )
    cfg.specificprogress[key] = "funasr模型运行中..." if web_l == "zh" else "funasr model is running..."
    res = cfg.funasr_model.generate(input=wav_file,
                                    batch_size_s=300,
                                    hotword='魔搭',
                                    device=device,
                                    )
    return res

def funasr_mutifile(wav_file,device,key,web_l):
    inference_pipeline = pipeline_queue.get()
    cfg.specificprogress[key]="funasr pipeline 运行中..." if web_l == "zh" else "funasr pipeline running..."
    res = inference_pipeline(input=wav_file, batch_size_s=300, hotword='魔搭', device=device, cache={})
    pipeline_queue.put(inference_pipeline)
    return res

import requests
import json
def qianfan_get_access_token(API_Key, Secret_Key):
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_Key}&client_secret={Secret_Key}"
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def qianfan_access_token_test(API_Key, Secret_Key):
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_Key}&client_secret={Secret_Key}"
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    while 1:
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            return response.json()
        except Exception as e:
            continue

def gemini_access_token_test(GEMINI_API_KEY):
    GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro?key=' + GEMINI_API_KEY
    while 1:
        try:
            response = requests.get(GEMINI_API_URL)
            return response.json()
        except Exception as e:
            continue

import numpy as np

def funasr_speaker_map(res, key, language):
    txt_spk_mapping=[]
    result_texts=[]
    spk = "说话者" if "zh" in language else "Speaker"
    for i, wrd_dict in enumerate(res[0]['sentence_info']):
        if not txt_spk_mapping:
            txt_spk_mapping.append({"text": [wrd_dict["text"]], "speaker": wrd_dict["spk"]})
        elif txt_spk_mapping and txt_spk_mapping[-1]["speaker"] == wrd_dict["spk"]:
            txt_spk_mapping[-1]["text"].append(wrd_dict["text"])
        else:
            result_texts.append(
                spk + str(txt_spk_mapping[-1]["speaker"]) + "：" + "".join(txt_spk_mapping[-1]["text"]))
            txt_spk_mapping.append({"text": [wrd_dict["text"]], "speaker": wrd_dict["spk"]})
    result_texts.append(spk + str(txt_spk_mapping[-1]["speaker"]) + "：" + "".join(txt_spk_mapping[-1]["text"]))
    return txt_spk_mapping, result_texts

def funasr_nospk(res, key):
    result_texts=[]
    for i, wrd_dict in enumerate(res[0]['sentence_info']):
        text = wrd_dict["text"]
        result_texts.append(text)
    return result_texts

def funasr_json(res, key):
    result_texts=[]
    for i, wrd_dict in enumerate(res[0]['sentence_info']):
        start = int(wrd_dict["start"])
        end = int(wrd_dict["end"])
        startTime = tool.ms_to_time_string(ms=start)
        endTime = tool.ms_to_time_string(ms=end)
        text = wrd_dict["text"]
        result_texts.append(
            {"line": len(result_texts) + 1, "start_time": startTime, "end_time": endTime, "text": text})
    return result_texts

def funasr_srt(res, key):
    result_texts = []
    for i, wrd_dict in enumerate(res[0]['sentence_info']):
        start = int(wrd_dict["start"])
        end = int(wrd_dict["end"])
        startTime = tool.ms_to_time_string(ms=start)
        endTime = tool.ms_to_time_string(ms=end)
        text = wrd_dict["text"]
        result_texts.append(f'{len(result_texts) + 1}\n{startTime} --> {endTime}\n{text}\n')
    return result_texts

def whisper_nospk(segments, key, language):
    result_texts=[]
    if language == "zhs":
        converter = opencc.OpenCC('t2s.json')
    else:
        converter = opencc.OpenCC('s2t.json')
    for segment in segments:
        text = converter.convert(segment.text) if "zh" in language else segment.text
        result_texts.append(text)
    return result_texts

def whisper_json(segments, key, language):
    result_texts=[]
    if language == "zhs":
        converter = opencc.OpenCC('t2s.json')
    else:
        converter = opencc.OpenCC('s2t.json')
    for segment in segments:
        start = int(segment.start * 1000)
        end = int(segment.end * 1000)
        startTime = tool.ms_to_time_string(ms=start)
        endTime = tool.ms_to_time_string(ms=end)
        text = converter.convert(segment.text) if "zh" in language else segment.text
        result_texts.append({"line": len(result_texts) + 1, "start_time": startTime, "end_time": endTime, "text": text})
    return result_texts

def whisper_srt(segments, key, language):
    result_texts=[]
    if language == "zhs":
        converter = opencc.OpenCC('t2s.json')
    else:
        converter = opencc.OpenCC('s2t.json')
    for segment in segments:
        start = int(segment.start * 1000)
        end = int(segment.end * 1000)
        startTime = tool.ms_to_time_string(ms=start)
        endTime = tool.ms_to_time_string(ms=end)
        text = converter.convert(segment.text) if "zh" in language else segment.text
        result_texts.append(f'{len(result_texts) + 1}\n{startTime} --> {endTime}\n{text}\n')
    return result_texts

import re
from deepmultilingualpunctuation import PunctuationModel

def whisper_speaker_map(segments, spk_ts, language, key, spkdi):
    def calculate_overlaps(ws, we, ss, se):
        return np.maximum(0, np.minimum(we, se) - np.maximum(ws, ss))
    txt_spk_mapping = []
    result_texts=[]
    spk_ts_np = np.array(spk_ts)
    ss = spk_ts_np[:, 0]
    se = spk_ts_np[:, 1]
    sp = spk_ts_np[:, 2]
    head = ""
    if language == "zhs":
        converter = opencc.OpenCC('t2s.json')
    else:
        converter = opencc.OpenCC('s2t.json')
    selected_speaker = spk_ts[0][-1]
    spk = "说话者" if "zh" in language else "Speaker"
    if spkdi == "punc":
        if "zh" in language:
            model = AutoModel(model="models/ct-punc-c")
        elif language in punct_model_langs:
            model = PunctuationModel(model="models/kredor-punctuate-all")
    for segment in segments:
        ws, we, txt = (
            int(segment.start * 1000),
            int(segment.end * 1000),
            converter.convert(segment.text) if "zh" in language else segment.text,  #中繁转化
        )
        overlaps = calculate_overlaps(ws, we, ss, se)
        total_overlaps = {}
        for speaker in np.unique(sp):
            indexs = np.where(sp == speaker)[0]
            total_overlaps[speaker] = overlaps[indexs].sum()
        if max(total_overlaps.values()) != 0:
            selected_speaker = max(total_overlaps, key=total_overlaps.get)
        # else:
        #     print(txt)
        if not txt_spk_mapping:
            txt_spk_mapping.append({"text": [txt], "start_time": ws, "end_time": we, "speaker": selected_speaker})
        elif txt_spk_mapping and txt_spk_mapping[-1]["speaker"] == selected_speaker:
            txt_spk_mapping[-1]["text"].append(txt)
            txt_spk_mapping[-1]["end_time"] = we
        else:
            if spkdi == "punc" and "zh" in language:
                texts = re.sub(r'[,?。、.…]', '', "".join(txt_spk_mapping[-1]["text"]))
                texts_with_punc = model.generate(input=texts)[0]["text"]
            elif spkdi == "punc" and language in punct_model_langs:
                # texts_with_punc="".join(txt_spk_mapping[-1]["text"])
                texts = re.sub(r'[,?。、.…]', '', "".join(txt_spk_mapping[-1]["text"]))
                texts_with_punc = model.restore_punctuation(texts)
            else:
                texts_with_punc = "".join(txt_spk_mapping[-1]["text"])

            result_texts.append(spk + str(txt_spk_mapping[-1]["speaker"]) + "：" + texts_with_punc)
            txt_spk_mapping.append({"text": [txt], "start_time": ws, "end_time": we, "speaker": selected_speaker})
    if spkdi == "punc" and "zh" in language:
        texts = re.sub(r'[,?。、.]', '', "".join(txt_spk_mapping[-1]["text"]))
        texts_with_punc = model.generate(input=texts)[0]["text"]
    elif spkdi == "punc" and language in punct_model_langs:
        # texts_with_punc = "".join(txt_spk_mapping[-1]["text"])
        texts = re.sub(r'[,?。、.]', '', "".join(txt_spk_mapping[-1]["text"]))
        texts_with_punc = model.restore_punctuation(texts)
    elif spkdi == "punc":
        texts_with_punc = "".join(txt_spk_mapping[-1]["text"])
        head = f"Punctuation restoration is not available for {language} language. Using the original punctuation.\n"
    else:
        texts_with_punc = "".join(txt_spk_mapping[-1]["text"])
    result_texts.append(spk + str(txt_spk_mapping[-1]["speaker"]) + "：" + texts_with_punc)
    return txt_spk_mapping, result_texts, head

def ernie_speed_128k_summary(record_texts, qianfan_apikey, qianfan_secretkey, language):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token=" + qianfan_get_access_token(qianfan_apikey, qianfan_secretkey)
    if "zh" in language:
        input = f'''\
        #### 目标

        请你阅读以下用<>括起来的语音转文本记录，并判断该文本记录对应的音频类型（例如：会议录音、辩论录音、访谈录音、讲座录音、语音笔记、新闻广播、播客、采访录音、日常生活录音等）。接着，分析用户可能希望从该文本记录中得到什么样的总结（例如：会议音频可能需要总结“待办事项”、“每个人的发言总结“和“会议重点”，辩论音频可能需要总结“不同论点陈述及论据总结”、“不同辩手的发言“等）。最后以JSON的形式返回你的分析结果。

        #### 步骤

        1. 首先，你需要仔细阅读和理解这段文本记录的内容，分析这段内容属于什么类型的录音。
        2. 然后，根据你对文本内容的理解以及你给出的音频属于的类型，思考如果你是音频的拥有者你对该音频的总结需求有哪些，以及每条总结需求所需的总结详细程度。
        3. 最后，以JSON格式返回你的分析结果。

        #### 注意

        1. 在分析时，确保你全面理解文本记录的内容、情景和讨论的主题。
        2. 请不要编造文本记录中未提及的内容。
        3. 总结需求需包含每条总结需求及每条需求所需的总结详细程度
        4. 你的分析结果应该包含两个字段：'音频类型'(audio_type)和'总结需求'(summary_requirements)。

        #### 返回格式

        请以如下JSON格式返回结果：

        {{
        "audio_type": "在此填写音频类型",
          "summary_requirements": "在此填写总结需求"
        }}

        #### 语音转文本记录

        <{record_texts}>
        '''
    else:
        input=f'''\
        #### Objective

        Read the following speech-to-text recordings enclosed in <> and determine what type of audio the recordings correspond to (e.g., meeting recordings, debate recordings, interview recordings, lecture recordings, voice notes, news broadcasts, podcasts, interview recordings, daily life recordings, etc.). Then, analyze what kind of summary the user may want to get from the text recording (e.g., meeting audio may need to summarize the "to-do list", "summary of everyone's speeches" and "highlights of the meeting", debate audio may need to summarize the "summary of the different arguments" and "statements by different debaters", etc.). Finally, return your analysis as JSON.

        #### Steps

        1. First, you need to carefully read and understand the content of this text recording, and analyze what type of recording it belongs to.
        2. Then, based on your understanding of the content of the text and the type of audio you have given, think about what summarization requirements you would have for the audio if you were the owner of the audio, and the level of detail required for each of the summarization requirements.
        3. Finally, return the results of your analysis in JSON format.

        #### Note

        1. When analyzing, make sure you fully understand the content, scenarios, and topics discussed in the text transcript.
        2. Do not make up anything that is not in the transcript.
        3. summarize the requirements by including each summary requirement and the level of detail required to summarize each requirement.
        4. Your analysis should contain two fields: 'audio_type' and 'summary_requirements'.

        #### Return Format

        Please return the results in the following JSON format:

        {{
        "audio_type": "Fill in the audio type here",
          "summary_requirements": "Fill in the summary requirements here"
        }}

        #### Speech-to-text recordings

        <{record_texts}>
        '''
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": input
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    while 1:
        try:
            response = requests.request("POST", url, headers=headers, data=payload).json().get("result")
            response = json.loads(response.replace('json', '').replace('```', ''))
            audio_type = response['audio_type']
            summary_requirements = response['summary_requirements']
            break
        except Exception as e:
            print(f"返回错误{e}，重新请求回答")
            continue
    if "zh" in language:
        input = f'''\
        #### 目标
        
        请你阅读以下用<>括起来的语音转文本记录，并总结这段语音转文本记录的内容。
        
        #### 步骤
        
        1. 首先，你需要仔细阅读和理解这段文本记录的内容。
        2. 据推测该音频类型为#{audio_type}#，用户的总结需求可能为#{summary_requirements}#，根据该总结需求以架构的格式对文本进行总结。
        3. 以markdown格式给出总结。
        
        #### 注意
        
        1. 你不会编造文本记录中未出现的事实与内容。
        2. 你只总结，不给出任何延伸。
        3. 总结中不会出现除文本记录外的内容，你的总结全部依赖于记录内容。
        4. 以需求作为标题，使用架构的格式给出总结，不遗漏需求。
        5. 以中文书写总结。
        
        #### 语音转文本记录
        
        <{record_texts}>
        '''
    else:
        input = f'''\
        #### Objectives
        
        Read the following speech-to-text transcript enclosed in <> and summarize the contents of this speech-to-text transcript.
        
        #### Steps
        
        1. First of all, you need to read and understand the content of this transcript carefully.
        2. It is assumed that the audio type is #{audio_type}# and the user's summarization requirement may be #{summary_requirements}#, according to this summarization requirement, summarize the text in the format of architecture. 3.
        3. give the summary in markdown format.
        
        #### Note
        
        1. you don't make up facts and content that don't appear in the text record.
        2. you summarize and do not give any extensions.
        3. nothing other than the text record appears in the summary; your summary depends entirely on the record.
        4. you use the requirements as headings and give the summary in an architectural format, without omitting requirements.
        5. Write the summary in English.
        
        #### Speech to text record
        
        <{record_texts}>
        '''
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": input
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    while 1:
        try:
            response = requests.request("POST", url, headers=headers, data=payload).json().get("result")
            return response
        except Exception as e:
            print(f"返回错误{e}，重新请求回答")
            continue

def gemini_flash_summary(record_texts, GEMINI_API_KEY, language):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=' + GEMINI_API_KEY
    if "zh" in language:
        input = f'''\
        #### 目标

        请你阅读以下用<>括起来的语音转文本记录，并判断该文本记录对应的音频类型（例如：会议录音、辩论录音、访谈录音、讲座录音、语音笔记、新闻广播、播客、采访录音、日常生活录音等）。接着，分析用户可能希望从该文本记录中得到什么样的总结（例如：会议音频可能需要总结“待办事项”、“每个人的发言总结“和“会议重点”，辩论音频可能需要总结“不同论点陈述及论据总结”、“不同辩手的发言“等）。最后以JSON的形式返回你的分析结果。

        #### 步骤

        1. 首先，你需要仔细阅读和理解这段文本记录的内容，分析这段内容属于什么类型的录音。
        2. 然后，根据你对文本内容的理解以及你给出的音频属于的类型，思考如果你是音频的拥有者你对该音频的总结需求有哪些，以及每条总结需求所需的总结详细程度。
        3. 最后，以JSON格式返回你的分析结果。

        #### 注意

        1. 在分析时，确保你全面理解文本记录的内容、情景和讨论的主题。
        2. 请不要编造文本记录中未提及的内容。
        3. 总结需求需包含每条总结需求及每条需求所需的总结详细程度
        4. 你的分析结果应该包含两个字段：'音频类型'(audio_type)和'总结需求'(summary_requirements)。

        #### 返回格式

        请以如下JSON格式返回结果：

        {{
        "audio_type": "在此填写音频类型",
          "summary_requirements": "在此填写总结需求"
        }}

        #### 语音转文本记录

        <{record_texts}>
        '''
    else:
        input = f'''\
        #### Objective

        Read the following speech-to-text recordings enclosed in <> and determine what type of audio the recordings correspond to (e.g., meeting recordings, debate recordings, interview recordings, lecture recordings, voice notes, news broadcasts, podcasts, interview recordings, daily life recordings, etc.). Then, analyze what kind of summary the user may want to get from the text recording (e.g., meeting audio may need to summarize the "to-do list", "summary of everyone's speeches" and "highlights of the meeting", debate audio may need to summarize the "summary of the different arguments" and "statements by different debaters", etc.). Finally, return your analysis as JSON.

        #### Steps

        1. First, you need to carefully read and understand the content of this text recording, and analyze what type of recording it belongs to.
        2. Then, based on your understanding of the content of the text and the type of audio you have given, think about what summarization requirements you would have for the audio if you were the owner of the audio, and the level of detail required for each of the summarization requirements.
        3. Finally, return the results of your analysis in JSON format.

        #### Note

        1. When analyzing, make sure you fully understand the content, scenarios, and topics discussed in the text transcript.
        2. Do not make up anything that is not in the transcript.
        3. summarize the requirements by including each summary requirement and the level of detail required to summarize each requirement.
        4. Your analysis should contain two fields: 'audio_type' and 'summary_requirements'.

        #### Return Format

        Please return the results in the following JSON format:

        {{
        "audio_type": "Fill in the audio type here",
          "summary_requirements": "Fill in the summary requirements here"
        }}

        #### Speech-to-text recordings

        <{record_texts}>
        '''
    body = {"contents": [{"role": "user", "parts": [{"text": input}]}]}
    headers = {"Content-Type": "application/json"}
    while 1:
        try:
            response = requests.post(url, data=json.dumps(body), headers=headers)
            result = response.json()
            response = result.get('candidates')[0].get('content', {}).get('parts', [])[0].get('text', '')
            response = json.loads(response.replace('json', '').replace('```', ''))
            audio_type = response['audio_type']
            summary_requirements = response['summary_requirements']
            break
        except Exception as e:
            print(f"返回错误{e}，重新请求回答")
            continue
    if "zh" in language:
        input=f'''\
        #### 目标
        
        请你阅读以下用<>括起来的语音转文本记录，并总结这段语音转文本记录的内容。
        
        #### 步骤
        
        1. 首先，你需要仔细阅读和理解这段文本记录的内容。
        2. 据推测该音频类型为#{audio_type}#，用户的总结需求可能为#{summary_requirements}#。
        3. 请根据上述信息对音频进行详细总结。
        
        #### 注意
        
        1. 在总结时，确保你全面理解文本记录的内容、情景和讨论的主题。
        2. 总结中不会出现除文本记录外的内容，你的总结全部依赖于记录内容。
        3. 请不要编造文本记录中未出现的事实与内容。
        4. 请不要添加任何意见、建议或延伸。
        5. 每个要点都应当清晰、简明地表达。
        6. 保证完整性，不遗漏重要信息。
        7. 以中文书写总结。
        
        #### 语音转文本记录
        
        <{record_texts}>
        '''
    else:
        input=f'''\
        #### Objectives
        
        Read the following speech-to-text transcript enclosed in <> and summarize the contents of this speech-to-text transcript.
        
        #### Steps
        
        1. First of all, you need to read and understand the content of this transcript carefully.
        2. It is presumed that the audio type is #{audio_type}# and the user's summarization requirements may be #{summary_requirements}#.
        3. Please summarize the audio in detail based on the above information.
        
        #### Note
        
        1. When summarizing, make sure you fully understand the content of the transcript, the scenario, and the topics discussed.
        2. There will be no content in the summary other than the transcript; your summary will rely entirely on the transcript.
        3. Do not make up facts and elements that do not appear in the transcript.
        4. Do not add any comments, suggestions, opinions or extensions.
        5. Each point should be expressed clearly and concisely.
        6. Ensure completeness and do not omit important information.
        7. Write summaries in English.
        
        #### Speech-to-text recordings
        
        <{record_texts}>
        '''
    body = {"contents": [{"role": "user", "parts": [{"text": input}]}]}
    headers = {"Content-Type": "application/json"}
    while 1:
        try:
            response = requests.post(url, data=json.dumps(body), headers=headers)
            result = response.json()
            response = result.get('candidates')[0].get('content', {}).get('parts', [])[0].get('text', '')
            return response
        except Exception as e:
            print(f"返回错误{e}，重新请求回答")
            continue