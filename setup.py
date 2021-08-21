

from setuptools import setup, find_packages

setup(
    name="fightingcv",
    version="0.0.2",
    author="xmu-xiaoma666",
    author_email="775629340@qq.com",
    description="External-Attention-pytorch -->公众号：FightingCV666",
    # 项目主页
    url="https://github.com/xmu-xiaoma666/External-Attention-pytorch", 
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages(),
    # python 版本
    python_requires='>=3',

    classifiers = [
            # 发展时期,常见的如下
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            # 开发的目标用户
            'Intended Audience :: Developers',
            # 属于什么类型
            'Topic :: Software Development :: Build Tools',
            # 许可证信息
            'License :: OSI Approved :: MIT License',
            # 目标 Python 版本
            'Programming Language :: Python :: 3.8',
        ],

    # # 安装过程中，需要安装的静态文件，如配置文件、service文件、图片等
    # data_files=[
    #     ('img', ['img/*']),
    #     ('analysis', ['analysis/*']),
    #            ],

    # # 希望被打包的文件
    # package_data={
    #     'attention':['attention/*.py'],
    #     'backbone_cnn':['backbone_cnn/*.py'],
    #     'conv':['conv/*.py'],
    #     'mlp':['mlp/*.py'],
    #     'rep':['rep/*.py'],
    #            },


    # 不打包某些文件
    exclude_package_data={
        'bandwidth_reporter':['*.md']
               },

    # install_requires 在安装模块时会自动安装依赖包
    extras_require={
        'numpy':['numpy==1.13.3']
    },


)
