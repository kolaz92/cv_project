import streamlit as st
import matplotlib.pyplot as plt    
import pickle as pkl

with open('history.pkl', 'rb') as file: # словарь оценок для графиков
        log = pkl.load(file)

st.write("# Метрики и графики модели предсказания погоды")
st.write(f'Accuracy train: {log['epoch_train_total_loss'][-1]:.4f}, Accuracy valid: {log['epoch_train_total_loss'][-1]:.4f}')

# зададим функцию рисования графиков
def plot_history(log, grid=True):
    fig, ax = plt.subplots(1,2, figsize=(14,5))

    ax[0].plot(log['epoch_train_total_loss'], label='train loss')
    ax[0].plot(log['epoch_valid_total_loss'], label='valid loss')
    ax[0].set_title(f'Loss on epoch {len(log['epoch_train_total_loss'])}')
    ax[0].grid(grid)
    ax[0].legend()


    ax[1].plot(log['epoch_train_accuracy'], label='train acc')
    ax[1].plot(log['epoch_valid_accuracy'], label='valid acc')
    ax[1].set_title(f'Accuracy on epoch {len(log['epoch_valid_total_loss'])}')
    ax[1].grid(grid)
    ax[1].legend()

    return fig

st.pyplot(plot_history(log))