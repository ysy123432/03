import streamlit as st
import numpy as np
import sys
import pickle
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from fastai.vision.all import *
import os




# 设置页面配置
st.set_page_config(
    page_title="ArtMatch | 画师推荐系统",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS
st.markdown("""
<style>
    /* 全局样式 */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #ec4899;
        --dark-color: #1f2937;
        --light-color: #f9fafb;
        --neutral-color: #e5e7eb;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    /* 头部样式 */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.1), 0 8px 10px -6px rgba(99, 102, 241, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        transform: translate(50%, -50%);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* 卡片样式 */
    .artist-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .artist-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }
    
    .artist-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
    }
    
    .artist-card:hover::before {
        transform: scaleX(1);
    }
    
    /* 风格标签样式 */
    .style-badge {
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-color);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .style-badge:hover {
        background-color: rgba(99, 102, 241, 0.2);
        transform: translateY(-1px);
    }
    
    /* 价格标签样式 */
    .price-tag {
        color: #10b981;
        font-weight: 600;
        font-size: 1.1rem;
        background-color: rgba(16, 185, 129, 0.08);
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
    }
    
    /* 相似度进度条 */
    .similarity-bar {
        height: 6px;
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 3px;
        margin-top: 0.75rem;
        overflow: hidden;
    }
    
    .similarity-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 3px;
        transition: width 1s ease-in-out;
    }
    
    /* 信息卡片 */
    .info-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* 统计卡片 */
    .stat-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 60px;
        height: 60px;
        background: rgba(99, 102, 241, 0.05);
        border-radius: 0 0 0 100%;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: rgba(0, 0, 0, 0.6);
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary-color);
        border-bottom: 3px solid var(--primary-color);
        font-weight: 600;
    }
    
    /* 滑块样式 */
    .stSlider [role="slider"] {
        background-color: var(--primary-color);
    }
    
    /* 选择框样式 */
    .stSelectbox > div {
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div:hover {
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
    }
    
    /* 进度指示器 */
    .progress-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .progress-circle {
        width: 30px;
        height: 30px;
        border: 3px solid rgba(99, 102, 241, 0.2);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
    
    /* 过渡动画 */
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* 数字动画 */
    .count-animation {
        display: inline-block;
        overflow: hidden;
        vertical-align: bottom;
    }
    
    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# 读取画师表格
try:
    artist_df = pd.read_excel(pathlib.Path(__file__).parent / "画师.xlsx")
except FileNotFoundError:
    st.error("未找到画师.xlsx 文件，请确保该文件与脚本在同一目录下。")
    st.stop()
# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

# 版本兼容
temp = None
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


# 读取模型
@st.cache_resource
def load_model_data():
    model_path = pathlib.Path(__file__).parent / "推荐.pkl"
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data
@st.cache_resource
def load_image_model_data():
    model_path = pathlib.Path(__file__).parent / "图片识别.pkl"
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

# def load_model():
#     """加载并缓存模型"""
#     # Windows 路径兼容性处理
#     temp = None
#     if sys.platform == "win32":
#         temp = pathlib.PosixPath
#         pathlib.PosixPath = pathlib.WindowsPath
    
#     try:
#         #model_path = pathlib.Path(__file__).parent / "图片识别.pkl"
#         # 使用 os.path.join 来确保路径兼容性
#         model_path = os.path.join(os.path.dirname(__file__), "图片识别.pkl")
#         model = load_learner(model_path)
#     finally:
#         # 恢复原始设置
#         if sys.platform == "win32" and temp is not None:
#             pathlib.PosixPath = temp
    
#     return model
# 加载模型数据
model_data = load_model_data()
artist_ids = model_data['artist_id']
style_vectors = np.array(model_data['style_vector'])
style_lists = model_data['style_list']
avg_prices = np.array(model_data['avg_price'])
similarity = model_data['similarity']

# 获取所有风格类型
all_styles = sorted({style for styles in style_lists for style in styles})

# 主标题区域
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>🎨 ArtMatch 画师推荐系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.9;'>根据您的喜好，精准匹配风格与预算的优质画师</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 创建两列布局
col1, col2 = st.columns([1, 2])

# 左侧边栏 - 用户选择
with col1:
    st.markdown("### 🎨 偏好设置")
    
    # 风格选择
    st.markdown("#### 选择您喜欢的风格")
    selected_styles = st.multiselect(
        "", 
        all_styles,
        placeholder="点击选择风格..."
    )
    
    # 价格范围
    st.markdown("#### 设置预算范围")
    price_range = st.slider(
        "", 
        min_value=0, 
        max_value=2000, 
        value=(200, 800), 
        step=50,
        format="¥%d"
    )
    
    # 排序选项
    st.markdown("#### 排序方式")
    sort_option = st.radio(
        "",
        ["相似度优先", "价格从低到高", "价格从高到低"],
        index=0
    )
    
    # 推荐数量
    st.markdown("#### 显示数量")
    num_recommendations = st.slider(
        "",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    # 推荐小贴士
    st.markdown("### 💡 推荐小贴士")
    tips = [
        "选择更多风格可以获得更广泛的推荐结果",
        "调整价格范围可以找到更符合预算的画师",
        "点击标签页查看推荐结果的详细分析",
        "刷新页面获取最新的画师数据"
    ]
    
    for i, tip in enumerate(tips):
        st.markdown(f"👉 **{tip}**")
        if i < len(tips) - 1:
            st.markdown("")

# 右侧主内容区 - 推荐结果
with col2:
    if selected_styles:
        # 显示加载动画
        with st.spinner("正在为您寻找匹配的画师..."):
            # 计算匹配分数
            match_scores = []
            for idx, styles in enumerate(style_lists):
                overlap = len(set(styles) & set(selected_styles))
                if overlap > 0:
                    match_scores.append((idx, overlap))
            match_scores.sort(key=lambda x: x[1], reverse=True)

            # 根据风格匹配结果进行价格过滤
            filtered = []
            for idx, score in match_scores:
                price = avg_prices[idx]
                if price_range[0] <= price <= price_range[1]:
                    filtered.append((artist_ids[idx], style_lists[idx], price, score))
            
            # 根据排序选项排序
            if sort_option == "价格从低到高":
                filtered.sort(key=lambda x: x[2])
            elif sort_option == "价格从高到低":
                filtered.sort(key=lambda x: x[2], reverse=True)
        
        # 结果展示
        st.markdown(f"### 🎯 为您找到 {len(filtered)} 位匹配画师")
        
        # 标签页
        tabs = st.tabs(["推荐列表", "使用指南"])
        
        # 推荐列表标签页
        with tabs[0]:
            if filtered:
                # 分批加载动画
                batch_size = 3
                num_batches = (len(filtered[:num_recommendations]) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(filtered[:num_recommendations]))
                    
                    # 为每批添加淡入动画
                    with st.container():
                        for artist_id, styles, price, score in filtered[start_idx:end_idx]:
                            # 查找所属平台
                            artist_info = artist_df[artist_df['画师ID'] == artist_id]
                            if not artist_info.empty:
                                platform = artist_info['所属平台'].values[0]
                                nickname = artist_info['画师昵称'].values[0]
                            else:
                                platform = "未知"
                                nickname = "未知"
                            st.markdown(f"""
                            <div class="artist-card fade-in">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <h3 style="color: #4f46e5; font-size: 1.2rem;">画师 ID: {artist_id}</h3>
                                    <span class="price-tag">¥{price}</span>
                                </div>
                                <div style="margin-bottom: 1rem;">
                                    <h4 style="margin-bottom: 0.5rem; font-size: 1rem; color: #4b5563;">所属平台: {platform}</h4>
                                    <h4 style="margin-bottom: 0.5rem; font-size: 1rem; color: #4b5563;">画师昵称: {nickname}</h4>
                                    <h4 style="margin-bottom: 0.5rem; font-size: 1rem; color: #4b5563;">风格:</h4>
                                    {''.join([f'<span class="style-badge">{style}</span>' for style in styles])}
                                </div>
                                <div>
                                    <h4 style="margin-bottom: 0.5rem; font-size: 1rem; color: #4b5563;">相似度: {score}/{len(selected_styles)}</h4>
                                    <div class="similarity-bar">
                                        <div class="similarity-fill" style="width: {score/len(selected_styles)*100}%"></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # 每批之间添加延迟
                    if batch_idx < num_batches - 1:
                        time.sleep(0.3)
            else:
                st.warning("没有找到满足条件的画师，请尝试更改筛选条件。")
                
                # 添加数据标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.5,
                        f'{int(height)}',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='500'
                    )
                
                plt.title("匹配风格分布", fontsize=14, fontweight='600')
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.ylabel("匹配数量", fontsize=11)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                st.pyplot(fig)
                
                # 价格分布
                prices = [price for _, _, price, _ in filtered]
                fig, ax = plt.subplots(figsize=(10, 6))
                
 
        
        # 使用指南标签页
        with tabs[1]:
            st.markdown("### 📖 使用指南")
            st.markdown("""
            **ArtMatch 画师推荐系统** 帮助您根据个人偏好找到最匹配的画师。以下是使用方法：
            
            1. **选择风格偏好**：从左侧选择您喜欢的一个或多个绘画风格
            2. **设置预算范围**：调整滑块以设定您的预期价格区间
            3. **选择排序方式**：根据相似度或价格进行排序
            4. **查看推荐结果**：系统将显示符合条件的画师列表
            
            每个推荐结果包含：
            - 画师ID：唯一标识符
            - 风格标签：点击查看更多详情
            - 价格信息：该画师的平均价格
            - 相似度指标：与您选择风格的匹配程度
            
            **提示**：
            - 选择更多风格可以获得更广泛的推荐
            - 调整价格范围可以找到更符合预算的画师
            - 点击"数据统计"标签页查看推荐结果的详细分析
            """)
            
            st.markdown("### 💡 推荐技巧")
            st.markdown("""
            - 如果您不确定自己喜欢什么风格，可以选择多种风格进行尝试
            - 如果没有找到满意的结果，尝试扩大价格范围或增加风格选择
            - 查看"数据统计"标签页中的价格分布，了解市场行情
            """)
            
            # 常见问题
            st.markdown("### ❓ 常见问题")
            faq_items = [
                ("如何获取更准确的推荐？", "尽量选择您真正喜欢的风格，而非所有风格。专注的选择会带来更精准的结果。"),
                ("价格是固定的吗？", "不是，价格是每位画师的平均价格，实际价格可能会因作品复杂度而有所不同。"),
                ("如何联系推荐的画师？", "目前系统仅提供推荐功能，您可以通过平台提供的联系方式与画师取得联系。"),
                ("为什么有些风格没有显示？", "可能是因为没有画师符合您的价格范围或选择的风格组合。"),
                ("系统多久更新一次数据？", "我们每天都会更新平台上的画师数据，确保您获得最新的推荐结果。")
            ]
            
            for question, answer in faq_items:
                with st.expander(question):
                    st.write(answer)
    else:
        st.markdown("### 👋 欢迎使用 ArtMatch")
        st.markdown("""
        请在左侧选择您喜欢的绘画风格和预算范围，系统将为您推荐匹配的优质画师。
        
        ArtMatch 采用先进的匹配算法，结合风格相似度和价格因素，为您提供精准的推荐结果。
        """)
        
        # # 示例风格推荐
        # st.markdown("### 🔥 热门风格推荐")
        
        # # 创建三列布局
        # col1, col2, col3 = st.columns(3)
        
        # with col1:
        #     st.markdown("#### 头像")
        #     st.image("https://huajia.fp.ps.netease.com/file/67910a23ad602dd1ffa878a5FErkHWva06", use_column_width=True, caption="细腻逼真的细节表现")
        #     st.markdown("写实主义风格注重细节和真实感，适合需要逼真效果的作品。")
        
        # with col2:
        #     st.markdown("#### 卡通风格")
        #     st.image("https://picsum.photos/seed/cartoon/300/200", use_column_width=True, caption="活泼有趣的人物形象")
        #     st.markdown("卡通风格以简化的线条和夸张的表现手法为特点，适合轻松活泼的内容。")
        
        # with col3:
        #     st.markdown("#### 插画")
        #     st.image("https://picsum.photos/seed/cyberpunk/300/200", use_column_width=True, caption="未来科技感的视觉冲击")
        #     st.markdown("赛博朋克风格融合未来科技与反乌托邦元素，营造出独特的视觉冲击力。")
        

        # 主应用
        st.title("画师推荐系统")
        st.write("上传一张图片，应用将根据您上传的图片推荐相应的画师。")

        model = load_model()

        uploaded_files = st.file_uploader("选择1张图片...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) == 1:
            img_file = uploaded_files[0]
            image = PILImage.create(img_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            recommend_button = st.button("用户推荐")
    
            if recommend_button:
                pred, pred_idx, probs = model.predict(image)
                # 查找画师信息
                artist_info = artist_df[artist_df['画师昵称'] == pred]
                if not artist_info.empty:
                    artist_id = artist_info['画师ID'].values[0]
                    nickname = artist_info['画师昵称'].values[0]
                    platform = artist_info['所属平台'].values[0]
                    st.write(f"画师 ID: {artist_id}")
                    st.write(f"画师昵称: {nickname}")
                    st.write(f"所属平台: {platform}")
                else:
                    st.write("未找到该画师的相关信息。")
                st.write(f"相似度: {probs[pred_idx]:.2%}")
        elif uploaded_files and len(uploaded_files) > 1:
            for img_file in uploaded_files:
                cols = st.columns(2)
                with cols[0]:
                    image = PILImage.create(img_file)
                    st.image(image, caption="上传的图片", use_container_width=True)
        
                with cols[1]:
                    pred, pred_idx, probs = model.predict(image)
                    # 查找画师信息
                    artist_info = artist_df[artist_df['画师昵称'] == pred]
                    if not artist_info.empty:
                        artist_id = artist_info['画师ID'].values[0]
                        nickname = artist_info['画师昵称'].values[0]
                        platform = artist_info['所属平台'].values[0]
                        st.write(f"画师 ID: {artist_id}")
                        st.write(f"画师昵称: {nickname}")
                        st.write(f"所属平台: {platform}")
                    else:
                        st.write("未找到该画师的相关信息。")
                    st.write(f"相似度: {probs[pred_idx]:.2%}")
                    st.divider()
            st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
        # 平台优势
    st.markdown("### ✨ 平台优势")
    features = [
            "精准匹配：基于深度学习的推荐算法",
            "多样化选择：超过50种绘画风格",
            "透明价格：明确的价格区间和参考",
            "优质画师：经过筛选的专业画师团队",
        ]
        
    for feature in features:
            st.markdown(f"✅ **{feature}**")



# 恢复路径类型（如果之前修改过）
if temp is not None:
    pathlib.PosixPath = temp
