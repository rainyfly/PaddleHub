### This script is used for translating Chinese Readme to English 
### PaddleHub module specific
import collections
import os
import re
import shutil

def analysis_module():
    '''
    fetch module names in modulename_all.txt
    '''
    modules_dict = {}
    with open('modulename_all.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            res = line.strip('\n')
            if res.startswith('#'):
                modules_dict[res[1:]] = []
                curkey = res[1:]
            else:
                modules_dict[curkey].append(res)
    return modules_dict

        

### Part 1 table translation
def translate_table():
    '''
    translate description table 
    '''
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    translation_dict = {
        '模型名称': 'Module Name',
        '类别': 'Category',
        '网络': 'Network',
        '数据集': 'Dataset',
        '是否支持Fine-tuning': 'Fine-tuning supported or not',
        '模型大小': 'Module Size',
        '最新更新日期': 'Latest update date',
        '数据指标': 'Data indicators'
    }
    words_dict = {
        '是': 'Yes',
        '否': 'No'
    }
    modules_dict = analysis_module()
    ### table pattern
    tablepattern = re.compile('\|(.+)\|(.+)\|')
    ###
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        match = tablepattern.search(sentense)
                        if match:
                            print(match, match.group(1), match.group(2))
                            key = match.group(1)
                            value = match.group(2)
                            if key in translation_dict:
                                sentense = sentense.replace(key, translation_dict[key])
                                if key == '类别':
                                    sentense = sentense.replace(value, category_dict[module_category])
                                elif key == '数据集':
                                    sentense = sentense.replace(value, value.replace('数据集', 'dataset'))
                                elif key == '是否支持Fine-tuning':
                                    sentense = sentense.replace(value, words_dict[value])
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)
                         
### Part 2 title translation
def translate_title():
    '''
    translate titles.
    '''
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}


    threesharp_translation_dict = {
        '应用效果展示': 'Application Effect Display',
        '模型介绍': 'Module Introduction',
        '环境依赖': 'Environmental Dependence',
        '安装': 'Installation',
        '命令行预测': 'Command line Prediction',
        '代码示例': 'Prediction Code Example',
        '预测代码示例': 'Prediction Code Example',
        '第一步：启动PaddleHub Serving': 'Step 1: Start PaddleHub Serving',
        '第二步：发送预测请求': 'Step 2: Send a predictive request',

    }

    twosharp_translation_dict = {
        '模型基本信息': 'Basic Information',
        '安装': 'Installation',
        '模型API预测': 'Module API Prediction',
        '服务部署': 'Server Deployment',
        '更新历史': 'Release Note',
        '一、': 'I.',
        '二、': 'II.',
        '三、': 'III.',
        '四、': 'IV.',
        '五、': 'V.'
    }

    onesharp_translation_dict = {
        '发送HTTP请求': 'Send an HTTP request',
        '打印预测结果': 'print prediction results'
    }
    modules_dict = analysis_module()
    ### table pattern
    titlepattern = re.compile('(#+) (.+)')
    ###
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        match = titlepattern.search(sentense)
                        if match:
                            print(match, match.group(1), match.group(2))
                            sharps = match.group(1)
                            numsharps = sharps.count('#')
                            if numsharps == 1:
                                for key, value in onesharp_translation_dict.items():
                                    if key in sentense:
                                        sentense = sentense.replace(key, value)
                            elif numsharps == 2:
                                for key, value in twosharp_translation_dict.items():
                                    if key in sentense:
                                        sentense = sentense.replace(key, value)
                            elif numsharps == 3:
                                for key, value in threesharp_translation_dict.items():
                                    if key in sentense:
                                        sentense = sentense.replace(key, value)
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)

### part 3 list item translation
def listitem_translation():
    '''
    translate listitem.
    '''
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    listitem_translation_dict = {
        '样例结果示例': 'Sample results',
        #'通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令]': 'If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction]', # special case
        '参数': 'Parameters',
        '返回': 'Return',
        '运行启动命令': 'Run the startup command',
        '**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置': '**NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set',
        #'默认端口号为8866': 'The servitization API is now deployed and the default port number is 8866.', # special case
        '配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果': 'With a configured server, use the following lines of code to send the prediction request and obtain the result',
        '。': '.'
        #'PaddleHub Serving 可以部署一个': 'PaddleHub Serving can deploy an online service of {}.' # special case
    }
    modules_dict = analysis_module()
    ### list item pattern
    listitem_pattern = re.compile('\- (.+)')
    ###
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        match = listitem_pattern.search(sentense)
                        if match:
                            # print(match, match.group(1))                         
                            for key, value in listitem_translation_dict.items():
                                    if key in sentense:
                                        sentense = sentense.replace(key, value)
                            if '默认端口号为8866' in sentense:
                                sentense = sentense.replace(match.group(1), 'The servitization API is now deployed and the default port number is 8866.')
                            if 'PaddleHub Serving 可以部署一个' in sentense:
                                sentense = sentense.replace(match.group(1), 'PaddleHub Serving can deploy an online service of {}.'.format(category_dict[module_category]))
                            if 'PaddleHub Serving可以部署一个' in sentense:
                                sentense = sentense.replace(match.group(1), 'PaddleHub Serving can deploy an online service of {}.'.format(category_dict[module_category]))   
                            if '通过命令行方式实现' in sentense:
                                sentense = sentense.replace(match.group(1), 'If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)')                    
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)

### part 4 version description translation
def versiondescription_translation():
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    description_translation_dict={
        '初始发布': 'First release',
        '修复numpy数据读取问题': 'Fix the problem of reading numpy',
        '适配paddlehub2.0': 'Adapt to paddlehub2.0',
        '删除batch_size选项': 'Delete optional parameter batch_size',
        '修复python2中编码问题': 'Fix the problem of encoding in python2',
        '提升预测性能以及易用性': 'Improve the prediction performance and users\' experience'      
    }
    modules_dict = analysis_module()
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        for key, value in description_translation_dict.items():
                            if key in sentense:
                                sentense = sentense.replace(key, value)
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)


### part 5 specific sentence translation
def specificsentence_translation():
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    install_helper_description = 'In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()'
    hubinstall_helper_description = '[How to install PaddleHub]()'
    modules_dict = analysis_module()
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        if '[如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)' in sentense:
                            sentense = sentense.replace('[如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)', hubinstall_helper_description)
                        if '**NOTE:** paths和images两个参数选择其一进行提供数据' in sentense:
                            sentense = sentense.replace('**NOTE:** paths和images两个参数选择其一进行提供数据', '**NOTE:** choose one parameter to provide data from paths and images')
                        if '如您安装时遇到问题' in sentense:
                            match = re.search('\- (.+)', sentense)
                            sentense = sentense.replace(match.group(1), install_helper_description)
                        if '零基础MacOS安装' in sentense:
                            continue
                        
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)

### part 6 api parameter translation
def api_parameter_translation():
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    general_apiparams_translation_table = {
        'paths (list\[str\])': 'paths (list[str]): image path;',
        'images (list\[numpy.ndarray\])': 'images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;',
        'use\_gpu (bool)': 'use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**',
        'use_gpu (bool)': 'use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**',
        'visualization (bool)': 'visualization (bool): Whether to save the results as picture files;',
        'output\_dir (str)': 'output_dir (str): save path of images;',
        'output_dir (str)': 'output_dir (str): save path of images;',
        'batch_size (int)': 'batch_size (int): the size of batch;',
        'batch\_size (int)': 'batch_size (int): the size of batch;',
    }
    image_classication_translation_table = {
        'data：dict类型': 'data (dict): key is "image", value is a list of image paths',
        'result：list类型': 'result(list[dict]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability',
        'top\_k (int)' : 'top\_k (int): return the first k results',
        'res (list\[dict\])': 'res (list\[dict\]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability'
    }

    face_detection_translation_table = {
        'shrink (float):' : 'shrink (float): the scale to resize image',
        'confs\_threshold (float):': 'confs\_threshold (float): the confidence threshold',
        'score_thresh (float):': 'score_thresh (float): the confidence threshold',
        'path (str):': 'path (str): path for input image',
        'data (list): 检测结果': 'data (list): detection results, each element in the list is dict',
        'confidence (float): 识别的置信度': 'confidence (float): the confidence of the result',
        'left (int): 边界框的左上角x坐标': "left (int): the upper left corner x coordinate of the detection box ",
        'top (int): 边界框的左上角y坐标': 'top (int): the upper left corner y coordinate of the detection box',
        'right (int): 边界框的右下角x坐标': 'right (int): the lower right corner x coordinate of the detection box',
        'bottom (int): 边界框的右下角y坐标': 'bottom (int): the lower right corner y coordinate of the detection box',
        'dirname: 存在模型的目录名称': 'dirname: output dir for saving model',
        'model\_filename: 模型文件名称': 'model\_filename: filename for saving model',
        'params\_filename: Parameters文件名称': 'params\_filename: filename for saving parameters',
        'combined: 是否将Parameters保存到统一的一个文件中.': 'combined: whether save parameters into one file',
        '将模型保存到指定路径' : 'Save model to specific path',
        '检测输入图片中的所有人脸位置.': 'Detect all faces in image'
    }

    objection_detection_translation_table = {
        '提取特征，用于迁移学习': 'Extract features, and do transfer learning',
        'trainable(bool): Parameters是否可训练': 'trainable(bool): whether parameters trainable or not',
        'pretrained (bool): 是否加载预训练模型': 'pretrained (bool): whether load pretrained model or not',
        'get\_prediction (bool): 是否执行预测': 'get\_prediction (bool): whether perform prediction',
        'inputs (dict): 模型的输入，keys 包括' : 'inputs (dict): inputs, a dict, include two keys: "image" and "im\_size"',
        'image (Variable): 图像变量': 'image (Variable): image variable',
        'im\_size (Variable): 图片的尺寸': 'im\_size (Variable): image size',
        'outputs (dict): 模型的输出' : 'outputs (dict): model output',
        'context\_prog (Program)': 'program for transfer learning',
        '预测API，检测输入图片中的所有目标的位置': 'Detection API, detect positions of all objects in image',
        'label (str): 标签': "label (str): label",
        'save\_path (str, optional): 识别结果的保存路径': 'save\_path (str, optional): output path for saving results'
    }
    description_translation_table={
        '分类接口API.': 'classification API.'
    }
    ### list item pattern
    apiitem_pattern = re.compile('\- (.+)')
    ###

    modules_dict = analysis_module()
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        match = apiitem_pattern.search(sentense)
                        if match:
                            # print(match, match.group(1))                         
                            for key, value in general_apiparams_translation_table.items():
                                if key in sentense:
                                    sentense = sentense.replace(match.group(1), general_apiparams_translation_table[key])
                            for key, value in image_classication_translation_table.items():
                                if key in sentense:
                                    sentense = sentense.replace(match.group(1), image_classication_translation_table[key])
                            for key, value in description_translation_table.items():
                                if key in sentense:
                                    sentense = sentense.replace(match.group(1), description_translation_table[key])
                            for key, value in face_detection_translation_table.items():
                                if key in sentense:
                                    sentense = sentense.replace(match.group(1), face_detection_translation_table[key])
                            for key, value in objection_detection_translation_table.items():
                                if key in sentense:
                                    sentense = sentense.replace(match.group(1), objection_detection_translation_table[key])
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)

##### part 7 batch process some modules, for temp use
def batchprocess():
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    classfication_resnet_vx_modules = ['resnet_v2_50_imagenet', 'resnet_v2_18_imagenet','resnet_v2_34_imagenet', 'resnet_v2_101_imagenet','resnet_v2_152_imagenet']
    modules_dict = analysis_module()
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        if 'the number of input and output channels is 4' in sentense:
                            sentense = sentense.replace('the number of input and output channels is 4', 'the number of input and output branch channels is 4')
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)


##### part 8 post process, for fixing something #########
def postprocess():
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    orig_install_helper_description = 'In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()'
    orig_hubinstall_helper_description = '[How to install PaddleHub]()'
    fix_install_helper_description = 'In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)'
    fix_hubinstall_helper_description = '[How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)'
    modules_dict = analysis_module()
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if not os.path.exists(os.path.join(module_path, 'README_en.md')):
                shutil.copy(os.path.join(module_path, 'README.md'), os.path.join(module_path, 'README_en.md'))
            oldreadme = os.path.join(module_path, 'README_en.md')
            newreadme = os.path.join(module_path, 'newREADME_en.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():
                        # if orig_install_helper_description in sentense:
                        #     sentense = sentense.replace(orig_install_helper_description, fix_install_helper_description)
                        # if orig_hubinstall_helper_description in sentense:
                        #     sentense = sentense.replace(orig_hubinstall_helper_description, fix_hubinstall_helper_description)     
                        # if  '预测Prediction Code Example' in sentense:
                        #     sentense = sentense.replace('预测Prediction Code Example', 'Prediction Code Example')              
                        if 'ndarray.shape 为' in sentense:
                            print('I am really bug:', oldreadme)
                            sentense = sentense.replace('ndarray.shape 为', 'ndarray.shape is ') 
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)


def temp_process():
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    modules_dict = analysis_module()
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            oldreadme = os.path.join(module_path, 'README.md')
            newreadme = os.path.join(module_path, 'newREADME.md')
            ### begin replace contents
            with open(oldreadme, 'r') as oldfile:
                with open(newreadme, 'w') as newfile:
                    for sentense in oldfile.readlines():  
                        if  '代码示例' in sentense:
                            sentense = sentense.replace('预测预测代码示例', '预测代码示例')                
                        newfile.write(sentense)
            shutil.move(newreadme, oldreadme)


def clearAll():
    '''
    delete all English README
    '''
    location_dict = {'目标检测':'modules/image/object_detection',
                     '人脸检测': 'modules/image/face_detection',
                     '深度估计': 'modules/image/depth_estimation',
                     '文字识别': 'modules/image/text_recognition',
                     '图像分类': 'modules/image/classification',
                     '图像生成': 'modules/image/Image_gan/style_transfer'}
    category_dict = {'目标检测': 'object detection',
                     '人脸检测': 'face detection',
                     '深度估计': 'depth estimation',
                     '文字识别': 'text recognition',
                     '图像分类': 'image classification',
                     '图像生成': 'image generation'}
    modules_dict = analysis_module()
    for module_category, module_names in modules_dict.items():
        if module_category not in location_dict:
            continue
        for module_name in module_names:
            module_path = os.path.join(location_dict[module_category], module_name)
            if os.path.exists(os.path.join(module_path, 'README_en.md')):
                os.remove(os.path.join(module_path, 'README_en.md'))
    


def main():
    # clearAll()
    # translate_table()
    # translate_title()
    # listitem_translation()
    # versiondescription_translation()
    # specificsentence_translation()
    # api_parameter_translation()
    # batchprocess()
    postprocess()
    # temp_process()
    


if __name__ == '__main__':
    main()

