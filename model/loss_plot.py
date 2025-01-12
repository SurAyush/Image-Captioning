import matplotlib.pyplot as plt
import numpy as np
import pickle

epoch = 4
train_loss = []
files = ['./loss_history/loss_hist_1_3000.pkl','./loss_history/loss_hist_1_6000.pkl','./loss_history/loss_hist_2_9000.pkl','./loss_history/loss_hist_1_12000.pkl',
         './loss_history/loss_hist_2_3000.pkl', './loss_history/loss_hist_2_6000.pkl', './loss_history/loss_hist_2_9000.pkl', './loss_history/loss_hist_2_12000.pkl',
         './loss_history/loss_hist_3_3000.pkl', './loss_history/loss_hist_3_6000.pkl', './loss_history/loss_hist_3_9000.pkl', './loss_history/loss_hist_3_12000.pkl',
         './loss_history/loss_hist_4_3000.pkl', './loss_history/loss_hist_4_6000.pkl', './loss_history/loss_hist_4_9000.pkl', './loss_history/loss_hist_4_12000.pkl']

loc = './loss_history/loss_hist_1_3000.pkl'

def process(loss_hist):
    output_list = [sum(loss_hist[i:i+200]) /200 for i in range(0, len(loss_hist), 200)]
    return output_list

for file in files:
    loss_hist = []
    with open(file, 'rb') as f:
        loss_hist = pickle.load(f)
        train_loss.extend(loss_hist)

train_loss = process(train_loss[:48400])
min_loss = min(train_loss)
iterations = np.arange(1, len(train_loss) + 1, 1)
iterations = iterations * 200

val_loss = []
with open('./loss_history/test_loss_hist.pkl','rb') as f:
    val_loss = pickle.load(f)

vloss = 0
for l in val_loss:
    vloss += l

vloss = vloss/len(val_loss)

plt.style.use('fast')  
plt.figure(figsize=(10, 6))  

plt.plot(iterations, train_loss, label='Training Loss', color='#007ACC', linewidth=1.5)
plt.title('Training Loss', fontsize=18)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)

plt.grid(alpha=0.5)

plt.legend(fontsize=12, loc='upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.annotate(
    f'Final Training Loss: {min_loss:.3f}\n Validation Loss: {vloss:.3f}\n (Total Epochs: {epoch})',
    xy = (iterations[-1], train_loss[-1]),  
    xytext = (iterations[-1] * 0.8, train_loss[-1] * 1.1),
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7)
)

plt.tight_layout() 
plt.show()
