from model import *

# train_cost, test_cost, train_acc, test_acc = train(epoch=100, batch_size=512, fix_len=32, lr=1e-5)
acc = evaluate(model_path='./model/best.ckpt', data_path='./data/test/')
print('test accuracy: '+str(acc))

