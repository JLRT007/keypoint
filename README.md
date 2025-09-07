# keypoint
## 安装环境 

```bash
###创建虚拟环境
conda create --name 环境名称 python=3.10.10
conda activate 环境名称
###看到安装成功即可，未安装成功再重新按步骤在安装一下
cd PaddleDetection
python setup.py install
pip install -r requirement.txt
###
cd ..
pip install -r requirements.txt
#再从paddlepaddle官网安装适合版本即可，我这里是cuda11.8
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

