1. 环境配置

   1. 首先，然后

      1. 首先在Anaconda的终端创建虚拟环境,并且启动虚拟环境

         ```
         conda create -n FDD python==3.7.16
         activate FDD
         ```

      2. 下包安装requirements的前面几包

         ```
         pip install opencv-contrib-python==4.2.0.34 -i https://pypi.tuna.tsinghua.edu.cn/simple
         pip install imutils==0.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
         pip install opencv-python==4.2.0.34 -i https://pypi.tuna.tsinghua.edu.cn/simple
         pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
         pip install cmake==3.18.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
         pip install scipy==1.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
         
         
         # 这一步需要目录所在位置是dlib-19.19.0-cp37-cp37m-win_amd64.whl所在的目录
         pip install dlib-19.19.0-cp37-cp37m-win_amd64.whl
         ```

      3. 在pycharm的右下角选择FDD的解释器

2. 在终端使用

   ```
   python FDD.py -p"video_1.avi" 
   ```

   ”video_1.avi“是视频文件名，（将数据集的视频拷贝出来验证比较快一点

3. 如果是疲劳返回yes，否则返回no

   