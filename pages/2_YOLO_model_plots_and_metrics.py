import streamlit as st
import matplotlib.pyplot as plt    
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

st.set_page_config(layout="wide")

st.write("# Метрики и графики модели детекции YOLO")

# Путь к изображениям
image_paths = [
    'exp3/confusion_matrix.png',  # замените на свои пути к изображениям
    'exp3/F1_curve.png',
    'exp3/P_curve.png',
    'exp3/R_curve.png',
    'exp3/PR_curve.png'
]

# Создание фигуры и сабплотов
fig, axes = plt.subplots(5, 1, figsize=(20, 20))  # 1 строка, num_images колонок

axes = axes.flatten()
# Проход по каждому изображению и его отображение
for ax, image_path in zip(axes, image_paths):
    if not image_path:
        ax.set_axis_off()
        continue    
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')  # Отключение осей

# Показ фигуры
plt.tight_layout()
st.pyplot(fig)

st.image(mpimg.imread('exp3/results.png'))
st.image(mpimg.imread('exp3/val_batch2_pred.jpg'))