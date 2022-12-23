import matplotlib.pyplot as plt

num_of_epochs1 = 2500
# num_of_epochs2 = 3000

overfit_loss = open('0.0 0.0 0.0 1.0 scheduler = True train.txt', 'r')

# loss = open('0.0 0.4 0.0 0.8 scheduler = True train_dropout0.2_with_aug.txt','r')

i=0
epochs1 =[]
loss1=[]
for i in range(num_of_epochs1):
    if i<3:
      garbage = overfit_loss.readline()
    else:
      line1 = overfit_loss.readline()
      words1 = line1.split()
      epochs1 += [int(words1[1])]
      loss1 += [float(words1[3])]

# j=0
# epochs2=[]
# loss2=[]
# for j in range(num_of_epochs2):
#     if j<3:
#       garbage = loss.readline()
#     else:
#       line2 = loss.readline()
#       words2 = line2.split()
#       epochs2 += [int(words2[1])]
#       loss2 += [float(words2[3])]

plt.plot(epochs1,loss1)
# plt.plot(epochs2,loss2)
plt.xlabel('number of epoch')
plt.ylabel('Loss')
plt.title("Loss functions")
plt.show()


overfit_loss.close()
