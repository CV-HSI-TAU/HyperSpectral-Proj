import matplotlib.pyplot as plt
import os

fig = plt.figure()
rows = 1
columns = 2
fig.add_subplot(rows, columns, 2)
for i in range(1,53,1):
    # mono_path = '/mnt/storage/datasets/khen_projects/adirido/Data_HS_Mono/Data Mono/' + str(i) + '/296.810mm.png'
    mono_path = '/mnt/storage/datasets/khen_projects/adirido/Data_HS_Mono/Data Mono/' + str(i) + '/'
    # hs_path = '/mnt/storage/datasets/khen_projects/adirido/Data_HS_Mono/Data HS/' + str(i) + '/75.png' mono_path = '/mnt/storage/datasets/khen_projects/adirido/Data_HS_Mono/Data Mono/' + str(i) + '/296.810mm.png'
    hs_path = '/mnt/storage/datasets/khen_projects/adirido/Data_HS_Mono/Data HS/' + str(i) +'/'
    mono_images = sorted(os.listdir(mono_path),
                         key=lambda x: os.path.splitext(x)[0])  # lists all the file that are in that folder
    hs_images = sorted(os.listdir(hs_path), key=lambda x: int(os.path.splitext(x)[0]))

    print(i,len(mono_images),len(hs_images))
    # mono = plt.imread(mono_path)
    # hs = plt.imread(hs_path)

    # plt.subplot(1, 2, 1)
    # plt.imshow(mono)
    # plt.axis('off')
    # plt.title("Mono")
    # plt.subplot(1, 2, 2)
    # plt.imshow(hs)
    # plt.axis('off')
    # plt.title("HS")
    # plt.show()
