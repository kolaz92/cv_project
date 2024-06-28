import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pickle as pkl
from torchvision.models import resnet18, ResNet18_Weights, resnet50
import streamlit as st
from PIL import Image
import zipfile
import requests
from io import BytesIO
import shutil

st.set_page_config(
    page_title='Предсказание детекции/локализации картинок',
)

st.sidebar.success('Выберите нужную страницу')

st.write('# Предсказание детекции овощей моделью ResNet')
st.write('# Предсказание локализации животных моделей YOLO')

# Функция-обработчик для переключателя 1
def toggle_1():
    if st.session_state.toggle1:
        st.session_state.toggle2 = False

# Функция-обработчик для переключателя 2
def toggle_2():
    if st.session_state.toggle2:
        st.session_state.toggle1 = False

# Инициализация состояния переключателей, если оно еще не установлено
if 'toggle1' not in st.session_state:
    st.session_state.toggle1 = True
if 'toggle2' not in st.session_state:
    st.session_state.toggle2 = False
if 'ByURL' not in st.session_state:
    st.session_state['ByURL'] = False

# Отображение переключателей с назначением функций-обработчиков
st.checkbox("Resnet", key='toggle1', on_change=toggle_1)
st.checkbox("YOLO", key='toggle2', on_change=toggle_2)

class LocModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = resnet18(ResNet18_Weights)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        # фризим слои, обучать их не будем (хотя технически можно)
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        # for param in self.feature_extractor[7].parameters():
        #     param.requires_grad = True

        # задай классификационный блок
        self.clf = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(128, 3)
        )

        # задай регрессионный блок
        self.box = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, img):
        # задай прямой проход
        resnet_out = self.feature_extractor(img)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        pred_classes = self.clf(resnet_out)
        pred_boxes = self.box(resnet_out)
        #print(pred_classes.shape, pred_boxes.shape)
        return pred_classes, pred_boxes

url = st.text_input("Введите URL изображения:")

# Функция для первой страницы - Загрузка файла
def upload_img():

    uploaded_file = st.file_uploader("Загрузите изображение или архив (jpg или png)", type=["jpg", "jpeg", "png", "zip"])
    imlist = []
    if uploaded_file:
        if uploaded_file.name.endswith(("jpg", "jpeg", "png")):
            # Загрузка и отображение одного изображения
            image = Image.open(uploaded_file)
            #st.image(image, caption="Загруженное изображение", use_column_width=True)
            imlist.append(image)
        elif uploaded_file.name.endswith("zip"):
            # Загрузка и отображение изображений из архива
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                # Извлечение всех файлов из архива
                zip_ref.extractall("extracted_images")
                
                # Отображение всех изображений
                for file_name in zip_ref.namelist():
                    if file_name.endswith(("jpg", "jpeg", "png")):
                        image_path = f"extracted_images/{file_name}"
                        image = Image.open(image_path)
                        #st.image(image, caption=file_name, use_column_width=True)
                        imlist.append(image)
        if imlist:
            st.session_state['ByURL'] = False
    return imlist

def pred_true_pics(img, pred_class, pred_box):
    l = list(pred_box[0].numpy() * 227)
    coord = (l[0],l[1])

    ix2cls = {
        0 : 'cucumber', 
        1 : 'eggplant', 
        2 : 'mushroom' 
        }

    # Создайте фигуру и оси
    fig, ax = plt.subplots(1,2)

    ax = ax.flatten()

    # Показать изображение
    ax[0].imshow(img[0].permute(1,2,0))
    ax[0].set_axis_off()
    ax[0].set_title(f'Class {ix2cls[pred_class.argmax().item()]}')

    ax[1].imshow(img[0].permute(1,2,0))
    rect = patches.Rectangle(coord,width=l[2]-l[0],height=l[3] - l[1], linewidth=5, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)
    ax[1].set_axis_off()
    # Покажите результат
    st.pyplot(fig)

def reseffloc(imlist):
    # Загрузка обученной модели
    Pmodel = torch.load('model.pth')
    Pmodel.load_state_dict(torch.load('model_weights.pth',map_location=torch.device('cpu'))) # модель и веса
    
    Pmodel.eval()
    valid_transforms = T.Compose(
    [   
        T.Resize((227, 227)),
        T.ToTensor()
    ]
    )
    with torch.no_grad():
        for img in imlist:
            img = img.convert("RGB")
            image = valid_transforms(img)        # применение трансформаций
            image = image.unsqueeze(0)      # добавление дополнительной размерности для батча
            pred_class, pred_box = Pmodel(image)
            pred_true_pics(image,pred_class,pred_box)

def yololoc(imlist,t):

    @st.cache_resource
    def get_model(conf):
        model = torch.hub.load(
            # будем работать с локальной моделью в текущей папке
            repo_or_dir = './yolov5/',
            model = 'custom', 
            path='exp3/weights/best.pt', 
            source='local',
            force_reload=True
            )
        model.eval()
        model.conf = conf
        print('Model loaded')
        return model

    with st.spinner():
        model = get_model(t)

    results=None
    reslist = []
    lcol, rcol = st.columns(2)
    with lcol:
        st.write()
        for img in imlist:
            results = model(img)
            st.image(img)
            if results:
                reslist.append(results)

    if reslist:
        with rcol:
            for res in reslist:
                st.image(res.render())



with st.sidebar:
    t = st.slider('Model conf', 0., 1., .1)

imlist = upload_img()
if st.button("Скачать изображение по URL"):
    st.session_state['ByURL'] = True

if url and st.session_state['ByURL']:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption="Скачанное изображение", use_column_width=True)
        imlist = []
        imlist.append(image)
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при скачивании изображения: {e}")
else:
    st.warning("Пожалуйста, введите URL изображения.")

# st.header(st.session_state['name'])

# if st.button('Jane'):
#     st.session_state['model'] = 'Jane Doe'
#     st.rerun()

# if st.button('John'):
#     st.session_state['model'] = 'John Doe'
#     st.rerun()

if st.button('Предсказать:'):
    if st.session_state.toggle1:
        reseffloc(imlist)
    if st.session_state.toggle2:
        yololoc(imlist,t)