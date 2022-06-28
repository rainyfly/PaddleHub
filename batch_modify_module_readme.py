import os
import re
import shutil

def main():
    modules = ['modules/image/Image_editing/super_resolution/falsr_a/', 'modules/image/Image_editing/super_resolution/falsr_b/',
    'modules/image/Image_gan/attgan_celeba/', 'modules/image/Image_gan/cyclegan_cityscapes/',
    'modules/image/Image_gan/stargan_celeba/', 'modules/image/Image_gan/stgan_celeba/',
    'modules/image/Image_gan/style_transfer/ID_Photo_GEN/', 'modules/image/Image_gan/style_transfer/UGATIT_83w/',
    'modules/image/Image_gan/style_transfer/UGATIT_92w/', 'modules/image/Image_gan/style_transfer/animegan_v2_paprika_54/',
    'modules/image/Image_gan/style_transfer/animegan_v2_paprika_97/']

    for modulename in modules:
        filename = os.path.join(modulename, 'README.md')
        newfilename = os.path.join(modulename, 'NewREADME.md')

        with open(filename, 'r') as fp:
            with open(newfilename, 'w') as newfp:
                for line in fp.readlines():                      
                    if "代码示例" in line:
                        line = line.replace('代码示例', '预测代码示例')
                    newfp.write(line)
        shutil.move(newfilename, filename)
        
main()