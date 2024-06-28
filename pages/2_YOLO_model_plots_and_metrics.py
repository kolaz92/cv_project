import streamlit as st
import matplotlib.pyplot as plt    
import pickle as pkl

with open('hist_bird.pkl', 'rb') as file: # словарь оценок для графиков
        hist_dict = pkl.load(file)

st.write("# Метрики и графики модели предсказания птиц")
st.write(f'Accuracy train: {hist_dict["train_accs"][-1]:.4f}, Accuracy valid: {hist_dict["valid_accs"][-1]:.4f}')

# зададим функцию рисования графиков
def plot_history(history, grid=True):
    fig, ax = plt.subplots(1,2, figsize=(14,5))

    ax[0].plot(history['train_losses'], label='train loss')
    ax[0].plot(history['valid_losses'], label='valid loss')
    ax[0].set_title(f'Loss on epoch {len(history["train_losses"])}')
    ax[0].grid(grid)
    ax[0].legend()

    ax[1].plot(history['train_accs'], label='train acc')
    ax[1].plot(history['valid_accs'], label='valid acc')
    ax[1].set_title(f'Accuracy on epoch {len(history["train_losses"])}')
    ax[1].grid(grid)
    ax[1].legend()

    return fig

st.pyplot(plot_history(hist_dict))