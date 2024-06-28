import streamlit as st
import matplotlib.pyplot as plt    
import pickle as pkl

with open('history.pkl', 'rb') as file: # словарь оценок для графиков
        log = pkl.load(file)

st.set_page_config(layout="wide")

st.write("# Метрики и графики модели классификации и предсказания овощей")
st.write(f'Accuracy train: {log['epoch_train_accuracy'][-1]:.4f}, Accuracy valid: {log['epoch_valid_accuracy'][-1]:.4f}')

# зададим функцию рисования графиков
def plot_history(log, grid=True):
    fig, ax = plt.subplots(3,2, figsize=(20,20))

    ax = ax.flatten()

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

    ax[2].plot(log['epoch_train_clf_loss'], label='train')
    ax[2].plot(log['epoch_valid_clf_loss'], label='valid')
    ax[2].set_title(f'Clf loss on epoch {len(log['epoch_train_total_loss'])}')
    ax[2].grid(grid)
    ax[2].legend()


    ax[3].plot(log['epoch_train_reg_loss'], label='train')
    ax[3].plot(log['epoch_valid_reg_loss'], label='valid')
    ax[3].set_title(f'Reg loss on epoch {len(log['epoch_valid_total_loss'])}')
    ax[3].grid(grid)
    ax[3].legend()

    ax[4].plot(log['epoch_train_iou'], label='train')
    ax[4].plot(log['epoch_valid_iou'], label='valid')
    ax[4].set_title(f'IoU on epoch {len(log['epoch_valid_total_loss'])}')
    ax[4].grid(grid)
    ax[4].legend()

    ax[5].set_axis_off()

    return fig

st.pyplot(plot_history(log))