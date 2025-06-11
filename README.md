# ArtMatch 画师推荐系统

这是一个使用 FastAI 训练的图像分类应用，可以识别并推荐与上传图像风格匹配的画师。

## 功能特点

支持上传 JPG、JPEG、PNG 格式的图片
使用 FastAI 训练的深度学习模型进行风格分类
根据上传的图片推荐匹配的画师及其作品风格
显示预测结果及相似度置信度
提供多样化的画师推荐，基于风格和预算进行智能匹配

## 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
streamlit run "D:\推荐系统\streamlit\artist_recommendation_system (4) - 副本.py"
```

## 部署到 Streamlit Cloud

1. 将代码推送到 GitHub 仓库
2. 在 [Streamlit Cloud](https://streamlit.io/cloud) 创建新应用
3. 连接到你的 GitHub 仓库
4. 指定主文件：`"D:\推荐系统\streamlit\artist_recommendation_system (4) - 副本.py"`

## 注意事项

- 确保模型文件 `图片识别.pkl` 存在于项目根目录
- 模型文件较大，建议使用 Git LFS 管理 