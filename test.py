import pickle

with open('./model/loss_history/loss_hist_1_3000.pkl','rb') as f:
    loss_hist = pickle.load(f)

s1 = 0
for el in loss_hist:
    s1+=el

print((s1)/len(loss_hist))
    
with open('./model/loss_history/loss_hist_4_12000.pkl','rb') as f:
    loss_hist = pickle.load(f)

s2 = 0
for el in loss_hist:
    s2+=el
    
print(s2/len(loss_hist))

with open('./model/loss_history/test_loss_hist.pkl','rb') as f:
    loss_hist = pickle.load(f)

s3 = 0
for el in loss_hist:
    s3+=el

print(s3/len(loss_hist))
