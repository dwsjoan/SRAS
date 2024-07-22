[English](./README_EN.md)

# 语音识别与AI总结

可用于本地语音转文字、说话人分割及简易的AI总结，搭配web端操作界面。

> 
> 除初次使用下载模型及选择使用AI总结时，语音转录及说话人分割过程无需联网，全程本地运行
> 
> 可选择使用 fast-whipser 或 funasr 模型，用于识别音视频中的人声并转为文字
>
> 可选用说话人分割功能，使用 NeMo（搭配Whisper）或 cam++（搭配Funasr）模型
> 
> 通过 opencc 模型纠正 whisper 的中文识别结果中的简繁混合问题
> 
> 可选用标点添加模型 ct-punc(中文)、punctuate-all(外文) 修正whisper的中/外文识别结果中的标点
> 
> 可选择输出json格式、srt字幕带时间戳格式、纯文字格式
> 
> 可选择百度 ernie-speed-128k 或谷歌 gemini-1.5-flash 总结语音内容
>


# 源码部署(Linux/Mac/Window)

1. 推荐 python 3.10

2. 创建空目录，如 E:/SRAS，在这个目录下打开 cmd 窗口（在该目录的地址栏中输入 `cmd`, 然后回车），使用git拉取源码到当前目录 `git clone git@github.com:dwsjoan/sras.git`

3. 创建虚拟环境 `python -m venv srasenv`

4. 激活环境，win下命令 `%cd%/srasenv/scripts/activate`，linux和Mac下命令 `source ./srasenv/bin/activate`

5. 在该虚拟环境下安装依赖: `pip install -r requirements.txt`。如果希望支持cuda加速，继续执行代码 `pip uninstall -y torch`, `pip install torch --index-url https://download.pytorch.org/whl/cu121`

6. win下解压 ffmpeg.7z，将其中的`ffmpeg.exe`和`ffprobe.exe`放在项目目录下，linux 和 mac 从 [ffmpeg 官方网站](https://ffmpeg.org/download.html) 下载相应版本的 ffmpeg，解压缩后将 `ffmpeg` 和 `ffprobe` 二进制程序放在项目根目录下

7. [下载fast-whisper模型压缩包](https://github.com/jianchang512/stt/releases/tag/0.0)，base->large-v3识别效果越来越好，但所需计算机资源也更多，根据需要下载模型，下载后将压缩包里的文件夹放到项目根目录的 models 文件夹内

8. [下载punctuate-all模型](https://huggingface.co/kredor/punctuate-all/tree/main)，将所有模型文件下载到项目根目录的 models/kredor-punctuate-all 文件夹内

9. 下载funasr系列模型，[语音识别模型](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)的模型文件下载到项目根目录 models/paraformer-zh 文件夹内，
    [标点模型](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/files)的模型文件下载到 models/ct-punc-c 文件夹内，
    [语音端点检测模型](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/files)的模型文件下载到 models/fsmn-vad 文件夹内，
    [说话人分割模型](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files)的模型文件下载到 models/cam++ 文件夹内。

10. 第一次使用说话人分割功能时需科学联网，会下载nemo模型到本地

11. 整体模型文件放置架构如图

 ![image](/static/images/example.png)

12. 执行 `python start.py`，等待自动打开本地浏览器窗口

# 注意事项

1. CUDA安装参考[本链接](https://juejin.cn/post/7318704408727519270)。如果没有英伟达显卡或未配置好CUDA环境，请勿使用 large/large-v3 模型，可能导致内存耗尽死机
2. 显存不足8G时，尽量避免使用largev-3模型，尤其是视频大于20M时，否则可能显存不足而崩溃。
3. 如果在启用了cuda并且电脑已安装好了cuda环境情况下出现未执行完毕就闪退的情况，可能是需要安装和cuda匹配的cudnn。如果cudnn按照教程安装好了仍闪退，极大概率是GPU显存不足，可以改为使用medium模型。
4. 可选funasr串行还是并行，并行通过创建pipeline实现，每个pipeline实例占用2GB显存。可设置最大并行数量，默认设置为2，可按需修改set.ini中的`funasr_num_pipelines`，显存较小时谨慎选用并行funasr
5. funasr建议使用串行
6. 推荐选择gemini
7. 同时选中自动导出与独立导出时，仅自动导出各音频的独立识别结果，需点击`导出文本`按钮方可导出合并后文本结果
8. 使用的音频文件会自动保存在 static/tmp 目录下

# 后续

三个主要问题 
>1. 【语音转文本结果准确度问题】录音质量不高或模型识别出错可能导致语音中一些重要部分被识别错，进而导致之后总结信息也出错
>2. 【用户对音频的总结需求问题】不同的用户对于音频想要的总结点是不一样的。不同主题的语音的总结重点可能完全不同，比如会议音频可能更需要总结出”待办事项“”会议重点“，而辩论音频可能更偏向”不同论点陈述及论据总结“之类的。而且同一段音频，不同的用户想要的总结内容也不一样，比如毕设答辩，学生可能更想要总结出老师的建议，老师可能想要总结出学生的答辩内容
>3. 【AI总结时出现的幻觉问题】一个是AI生成的总结可能会修改一些原文本的内容，一个是AI可能会在总结里混一些自己的想法或者建议，难以判断哪些是总结哪些是AI给出的延伸

AI总结只接了两个免费的模型，后续可以加别的模型。或者可以加音频背景音与人声分离，来提高音频质量。

目前AI总结是用了两阶段，一阶段AI判断音频类型和音频的总结需求，二阶段AI根据类型和需求给出总结。但AI给出的具体需求点不一定是人想要的，且AI对每个需求点的重要程度判断也不一定和用户一样。
之后可能会修改，比如AI给出一些总结点，用户来选择或者添加，然后再总结；比如通过某种方法辅助筛选和修正总结中的错误或延伸内容。

欢迎交流，邮箱 j1301771092@gmail.com 或者直接加QQ 2200518834。

# 致谢

本项目主要依赖的项目

1. 前后端框架来自 [stt](https://github.com/jianchang512/stt)
2. nemo_process.py 来自 [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)
3. funasr并行化的方法来自 https://blog.csdn.net/huiguo_/article/details/134676719

本项目依赖的其他项目

1. https://github.com/MahmoudAshraf97/whisper-diarization
2. https://github.com/SYSTRAN/faster-whisper
3. https://github.com/modelscope/FunASR
4. https://github.com/NVIDIA/NeMo
5. https://github.com/BYVoid/OpenCC
6. https://huggingface.co/kredor/punctuate-all
7. https://github.com/pallets/flask
8. https://ffmpeg.org/
9. https://layui.dev
10. https://blog.csdn.net/huiguo_/article/details/134676719

