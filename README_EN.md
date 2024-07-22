[简体中文](./README.md) 

# Speech Recognition and AI Summary

Can be used for local speech to text, speaker segmentation and Simple AI summarization with web-based interface.
> 
> The speech transcription and speaker segmentation process runs locally without an Internet connection, except for the first time when you download the model and when you choose to use AI summarization
> 
> Optionally, fast-whipser or funasr models can be used to recognize human voices in audio and video and convert them to text
>
> Optional speaker segmentation using NeMo (with Whisper) or cam++ (with Funasr) model
> 
> Correct the Chinese mixing problem in whisper's Chinese recognition results by opencc modeling
> 
> Optional punctuation models ct-punc (for Chinese) and punctuate-all (for foreign languages) to correct punctuation in Chinese/foreign language recognition results of whisper
> 
> Output json format, srt subtitle with timestamp format, plain text format
>
> Choose Baidu ernie-speed-128k or Google gemini-1.5-flash to summarize speech content
>


# Source Code Deployment (Linux / Mac / Window)

1. Recommended python 3.10

2. Create an empty directory, such as E:/SRAS, open a cmd window in this directory (type `cmd` in the address bar of the directory, and then enter), and use git to pull the source code into the current directory `git clone git@github.com:dwsjoan/sras.git`

3. Create a virtual environment `python -m venv srasenv`

4. Activate the environment with the command `%cd%/srasenv/scripts/activate` under Win and `source . /srasenv/bin/activate`

5. Install the dependencies in this virtual environment: `pip install -r requirements.txt`. If you want to support cuda acceleration, continue with the code `pip uninstall -y torch`, `pip install torch --index-url https://download.pytorch.org/whl/cu121`

6. Unzip ffmpeg.7z under Win, put `ffmpeg.exe` and `ffprobe.exe` in the project directory, and for Linux and Mac, download the appropriate version of ffmpeg from the [ffmpeg official website](https://ffmpeg.org/download.html), unzip it, and put the `ffmpeg` and `ffprobe` binaries in the project root directory. ffmpeg` and `ffprobe` binaries in the project root directory

7. [Download fast-whisper model zip](https://github.com/jianchang512/stt/releases/tag/0.0), base->large-v3 recognition is getting better and better, but also need more computer resources, according to the need to download the model, download the zip package in the folder into the models folder in the project root directory

8. [Download punctuate-all models](https://huggingface.co/kredor/punctuate-all/tree/main), download all model files to the models/kredor-punctuate-all folder in the project root directory

9. Download the models of the funasr series, [speech recognition models](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404- The model files of [pytorch/files]() are downloaded to the root directory models/paraformer-zh folder.
    The model files of [punctuation model](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/files) are downloaded to the root directory models/ct-punc-c folder.
    The model files for [speech endpoint detection model](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/files) are downloaded to the root folder models/fsmn-vad.
    The model file of [speaker segmentation model](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files) is downloaded to the root directory models/cam++.

10. The first time you use the speaker segmentation feature, you need to be connected to the Internet, and the nemo model will be downloaded locally

11. The overall model file placement structure is shown in Figure

![image](/static/images/example.png)

12. Execute `python start.py ` and wait for a local browser window to open automatically

# Notices

1. CUDA installation reference [this link](https://juejin.cn/post/7318704408727519270). If you do not have a NVIDIA graphics card or have not configured the CUDA environment, please do not use the large/large-v3 model, as it may run out of memory and crash!
2. When the video memory is less than 8G, try to avoid using the largev-3 model, especially when the video is larger than 20M, otherwise it may crash due to insufficient video memory.
3. If you have cuda enabled and have installed the cuda environment, you may need to install a cudnn that matches cuda. if cudnn is still flashing after following the tutorials, there is a high probability that you do not have enough GPU memory, so you can use the medium model instead.
4. You can choose whether to run the funasr in serial or parallel. Parallelism is realized by creating pipelines, each pipeline instance occupies 2GB of video memory. Maximum number of parallel instances can be set, the default is 2, you can modify `funasr_num_pipelines` in set.ini as needed, be careful when using parallel funasr with small video memory.
5. Funasr recommends using serial
6. Recommended for gemini
7. When both automatic export and independent export are selected, only the independent recognition results of each audio will be automatically exported, and you need to click the `Export Text` button to export the merged text results
8. The audio files used are automatically saved in the static/tmp directory

# Follow-up

Three main problems 
>1. [Accuracy of Speech-to-Text Results] Poor recording quality or model recognition errors may lead to misrecognition of some important parts of the speech, which in turn leads to errors in summarizing the information afterwards.
>2. [Users' summarization needs for audio] Different users want different summarization points for audio. The summarization focus of different topics may be completely different, for example, conference audio may need to summarize “to-do list” and “meeting highlights”, while debate audio may be more inclined to “summary of different arguments and summarization of arguments” and so on. “and summarize different arguments. And the same audio, different users want to summarize the content is not the same, for example, Bishop defense, students may want to summarize the teacher's advice, the teacher may want to summarize the content of the student's defense!
>3. [Illusion problem when AI summarizes] One is that AI-generated summaries may modify some contents of the original text, and one is that AI may mix some of its own ideas or suggestions in the summary, and it is difficult to judge which is the summary and which is the extension given by AI

AI summarization is only connected to two free models, and other models can be added subsequently. Or add audio background sound separated from vocals to improve audio quality.

At present, AI summarization is used in two stages, one stage AI judges the type of audio and audio summary needs, and the second stage AI gives a summary according to the type and needs. However, the specific demand points given by AI are not necessarily what people want, and AI's judgment of the importance of each demand point is not necessarily the same as the user.
After that, it may be modified, for example, AI gives some summary points, the user to choose or add, and then summarize; for example, through a certain method to assist in filtering and correcting the summary of the error or extension of the content.

Welcome to exchange, e-mail j1301771092@gmail.com or directly add QQ account 2200518834.

# Acknowledgement

Projects on which this project is primarily dependent

1. The front- and back-end frameworks come from [stt](https://github.com/jianchang512/stt)
2. nemo_process.py comes from [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)
3. The solution to funasr parallelization comes from https://blog.csdn.net/huiguo_/article/details/134676719

Other projects on which the project depends

1. https://github.com/SYSTRAN/faster-whisper
2. https://github.com/modelscope/FunASR
3. https://github.com/NVIDIA/NeMo
4. https://github.com/BYVoid/OpenCC
5. https://huggingface.co/kredor/punctuate-all
6. https://github.com/pallets/flask
7. https://ffmpeg.org/
8. https://layui.dev
9. https://blog.csdn.net/huiguo_/article/details/134676719
