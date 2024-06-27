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
import os
import pickle as pkl
from torchvision.models import resnet18, ResNet18_Weights, resnet50
import streamlit as st
from PIL import Image
import zipfile
import io

st.set_page_config(
    page_title='Предсказание детекции/локализации картинок',
)

st.sidebar.success('Выберите нужную страницу')

st.write('# Предсказание детекции овощей моделью ResNet')
st.write('# Предсказание локализации животных моделей YOLO')

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





# def load_and_predict(img,IsWeather=True):
#     pred_type = 'weather' if IsWeather else 'bird'

#     # Загрузка обученной модели
#     Pmodel = myRegNet() if IsWeather else myResNet_50()
#     weights = f'model_weights_{pred_type}.pth' if IsWeather else f'model_weights_{pred_type}.pt'

#     Pmodel.load_state_dict(torch.load(weights,map_location=torch.device('cpu'))) # модель и веса
#     #st.write(type(torch.load(weights,map_location=torch.device('cpu'))))

#     Pmodel.eval()

#     with open(f'classes_{pred_type}.pkl', 'rb') as file: # словарь классов
#         class_to_idx = pkl.load(file)
#         class_to_idx = {value:key for key,value in class_to_idx.items()}

#     #st.write(class_to_idx)

#     class GrayToRGB(object):
#         def __call__(self, img):
#             if img.mode == 'L':
#                 img = img.convert('RGB')
#             return img
        
#     if IsWeather:
#         valid_transforms = T.Compose([
#             GrayToRGB(),
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     else:
#         valid_transforms = T.Compose([  
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def load_image(img): # загрузка изображения
#         image = valid_transforms(img)        # применение трансформаций
#         image = image.unsqueeze(0)      # добавление дополнительной размерности для батча
#         return image

#     def predict(img):
#         img = load_image(img)
#         with torch.no_grad():
#             output = Pmodel(img)
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
#         predicted_class = probabilities.argmax().item()
#         return predicted_class

#     class_prediction = predict(img)
#     st.write(f' ### Предсказанный класс: {class_prediction}, Название класса: {class_to_idx[class_prediction]}')

# Функция для первой страницы - Загрузка файла
def upload_img():
    st.title("Загрузите фотографию или архив фотографий")

    uploaded_file = st.file_uploader("Загрузите изображение или архив (jpg или png)", type=["jpg", "jpeg", "png", "zip"])
    if uploaded_file:
        if uploaded_file.name.endswith(("jpg", "jpeg", "png")):
            # Загрузка и отображение одного изображения
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
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
                        st.image(image, caption=file_name, use_column_width=True)

res = upload_img()

# Загрузка обученной модели
Pmodel = torch.load('model.pth')
Pmodel.load_state_dict(torch.load('model_weights.pth',map_location=torch.device('cpu'))) # модель и веса