import os
import re
from collections import Counter

def model_statistics():
    '''
    用于统计每个种类的模型数量
    '''
    image_dir = './image'
    audio_dir = './audio'
    text_dir = './text'
    video_dir = './video'
    industry_dir = './image'
    imagemodule_map = {
        'classification': os.path.join(image_dir, 'classification'),
        'Image_gan': os.path.join(image_dir, 'Image_gan'),
        'keypoint_detection': os.path.join(image_dir, 'keypoint_detection'),
        'semantic_segmentation': os.path.join(image_dir, 'semantic_segmentation'),
        'face_detection': os.path.join(image_dir, 'face_detection'),
        'text_recognition': os.path.join(image_dir, 'text_recognition'),
        'Image_editing': os.path.join(image_dir, 'Image_editing'),
        'instance_segmentation': os.path.join(image_dir, 'instance_segmentation'),
        'object_detection': os.path.join(image_dir, 'object_detection'),
        'depth_estimation': os.path.join(image_dir, 'depth_estimation')
    }
    textmodule_map = {
        'text_generation': os.path.join(text_dir, 'text_generation'),
        'embedding': os.path.join(text_dir, 'embedding'),
        'machine_translation': os.path.join(text_dir, 'machine_translation'),
        'language_model': os.path.join(text_dir, 'language_model'),
        'sentiment_analysis': os.path.join(text_dir, 'sentiment_analysis'),
        'syntactic_analysis': os.path.join(text_dir, 'syntactic_analysis'),
        'simultaneous_translation': os.path.join(text_dir, 'simultaneous_translation'),
        'lexical_analysis': os.path.join(text_dir, 'lexical_analysis'),
        'punctuation_restoration': os.path.join(text_dir, 'punctuation_restoration'),
        'text_review': os.path.join(text_dir, 'text_review')
    }   
    audiomodule_map = {
        'voice_cloning': os.path.join(audio_dir, 'voice_cloning'),
        'tts': os.path.join(audio_dir, 'tts'),
        'asr': os.path.join(audio_dir, 'asr'),
        'audio_classification': os.path.join(audio_dir, 'audio_classification')
    }
    videomodule_map = {
        'video_classification': os.path.join(video_dir, 'classification'),
        'Video_editing': os.path.join(video_dir, 'Video_editing'),
        'multiple_object_tracking': os.path.join(video_dir, 'multiple_object_tracking')
    }
    industrymodule_map = {
        'industrial_application': os.path.join(industry_dir, 'industrial_application')
    }
    summary_map = {
        'image' : imagemodule_map,
        'text': textmodule_map,
        'audio': audiomodule_map,
        'video': videomodule_map,
        'industry': industrymodule_map
    }
    counter = Counter()
    for category, modulemap in summary_map.items():
        for subcategory, directory in modulemap.items():
            for name in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, name)):
                    counter[subcategory] += 1
                    counter[category] += 1
    return counter
    
def analysis_README(filepath):
    '''
    解析README文件，返回模型表格所需要的module name, 网络，数据集名称
    '''
    results = {}
    ### table pattern
    tablepattern = re.compile('\|(.+)\|(.+)\|')
    if not os.path.exists(filepath):
        return results
    with open(filepath, 'r') as filehandle:
         for sentense in filehandle.readlines():
            match = tablepattern.search(sentense)
            if match:
                print(match, match.group(1), match.group(2))
                key = match.group(1)
                value = match.group(2)
                if key == '模型名称':
                    results['modulename'] = value
                if key == '网络':
                    results['network'] = value
                if key == '数据集':
                    results['dataset'] = value
    return results
    

def main():
    ### 类别名 - 目录路径 映射表, 该文件需要放在modules目录下运行，因为写的是相对路径
    image_dir = './image'
    audio_dir = './audio'
    text_dir = './text'
    video_dir = './video'
    industry_dir = './image'

    image_path_map = {
        # 图像
        '图像分类': os.path.join(image_dir, 'classification'),
        '图像生成': os.path.join(image_dir, 'Image_gan'),
        '关键点检测': os.path.join(image_dir, 'keypoint_detection'),
        '图像分割': os.path.join(image_dir, 'semantic_segmentation'),
        '人脸检测': os.path.join(image_dir, 'face_detection'),
        '文字识别': os.path.join(image_dir, 'text_recognition'),
        '图像编辑': os.path.join(image_dir, 'Image_editing'),
        '实例分割': os.path.join(image_dir, 'instance_segmentation'),
        '目标检测': os.path.join(image_dir, 'object_detection'),
        '深度估计': os.path.join(image_dir, 'depth_estimation')
    }
    text_path_map = {
        # 文本
        '文本生成': os.path.join(text_dir, 'text_generation'),
        '词向量': os.path.join(text_dir, 'embedding'),
        '机器翻译': os.path.join(text_dir, 'machine_translation'),
        '语义模型': os.path.join(text_dir, 'language_model'),
        '情感分析': os.path.join(text_dir, 'sentiment_analysis'),
        '句法分析': os.path.join(text_dir, 'syntactic_analysis'),
        '同声传译': os.path.join(text_dir, 'simultaneous_translation'),
        '词法分析': os.path.join(text_dir, 'lexical_analysis'),
        '标点恢复': os.path.join(text_dir, 'punctuation_restoration'),
        '文本审核': os.path.join(text_dir, 'text_review')
    }
    audio_path_map = {
        # 语音
        '声音克隆': os.path.join(audio_dir, 'voice_cloning'),
        '语音合成': os.path.join(audio_dir, 'tts'),
        '语音识别': os.path.join(audio_dir, 'asr'),
        '声音分类': os.path.join(audio_dir, 'audio_classification')
    }
    video_path_map = {
        # 视频
        '视频分类': os.path.join(video_dir, 'classification'),
        '视频修复': os.path.join(video_dir, 'Video_editing'),
        '多目标追踪': os.path.join(video_dir, 'multiple_object_tracking')
    }
    industry_path_map = {
        # 工业应用
        '表针识别': os.path.join(industry_dir, 'industrial_application')
    }
    summary_map = {
        '图像':  image_path_map,
        '文本':  text_path_map,
        '语音':  audio_path_map,
        '视频':  video_path_map,
        '工业应用': industry_path_map
    }

    ### 目录
    content = '''# 目录
|[图像](#图像) （{image}个）|[文本](#文本) （{text}个）|[语音](#语音) （{audio}个）|[视频](#视频) （{video}个）|[工业应用](#工业应用) （{industry}个）|
|--|--|--|--|--|
|[图像分类](#图像分类) ({classification})|[文本生成](#文本生成) ({text_generation})| [声音克隆](#声音克隆) ({voice_cloning})|[视频分类](#视频分类) ({video_classification})| [表针识别](#表针识别) ({industrial_application})|
|[图像生成](#图像生成) ({Image_gan})|[词向量](#词向量) ({embedding})|[语音合成](#语音合成) ({tts})|[视频修复](#视频修复) ({Video_editing})|-|
|[关键点检测](#关键点检测) ({keypoint_detection})|[机器翻译](#机器翻译) ({machine_translation})|[语音识别](#语音识别) ({asr})|[多目标追踪](#多目标追踪) ({multiple_object_tracking})|-|
|[图像分割](#图像分割) ({semantic_segmentation})|[语义模型](#语义模型) ({language_model})|[声音分类](#声音分类) ({audio_classification})| -|-|
|[人脸检测](#人脸检测) ({face_detection})|[情感分析](#情感分析) ({sentiment_analysis})|-|-|-|
|[文字识别](#文字识别) ({text_recognition})|[句法分析](#句法分析) ({syntactic_analysis})|-|-|-|
|[图像编辑](#图像编辑) ({Image_editing})|[同声传译](#同声传译) ({simultaneous_translation})|-|-|-|
|[实例分割](#实例分割) ({instance_segmentation})|[词法分析](#词法分析) ({lexical_analysis})|-|-|-|
|[目标检测](#目标检测) ({object_detection})|[标点恢复](#标点恢复) ({punctuation_restoration})|-|-|-|
|[深度估计](#深度估计) ({depth_estimation})|[文本审核](#文本审核) ({text_review})|-|-|-|
'''

    ### 表格
    table_title = '''
|module|网络|数据集|简介|
|--|--|--|--|
'''
    table_pattern = '|[{modulename}]({relativepath})|{network}|{dataset}|{introduction}|\n'
    ### 

    with open('modellist.md', 'w') as filehandle:
        ### 输出总目录
        counter = model_statistics()
        filehandle.write(content.format(**counter))
        filehandle.write('\n')
        ### 输出每个类别的模型汇总
        for category, modulemap in summary_map.items():
            filehandle.write('## {}\n'.format(category))
            for subcategory, directory in modulemap.items():
                filehandle.write('  - ### {}\n'.format(subcategory))
                filehandle.write(table_title)
                for name in os.listdir(directory):
                    if os.path.isdir(os.path.join(directory, name)):
                        information = analysis_README(os.path.join(directory, name, 'README.md'))
                        if not information: ## 该模块没有README.md文件
                            information['modulename'] = name
                            information['network'] = ''
                            information['dataset'] = ''
                        information['introduction'] = '' #暂时设置为空
                        information['relativepath'] = os.path.relpath(os.path.join(directory, name))
                        filehandle.write(table_pattern.format(**information))
                filehandle.write('\n')

if __name__ == '__main__':
    main()
                

                


        



                    
    