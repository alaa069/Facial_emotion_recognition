# Copyright (c) 2018 Alaa BEN JABALLAH

import net_training as nt

# datadir = "D:\\inzynierka\\data\\"
# imagedir = "cohn-kanade-images\\"
# labeldir = "Emotion\\"

network = nt.build_cnn('models/model.npz')

# nt.train_net(datadir, imagedir, labeldir, network)

faces = nt.load_img("images/anger.jpg")
tab = nt.evaluate(network, faces)
print("anger = {:.2f}%\ncontempt = {:.2f}%\ndisgust = {:.2f}%\nfear = {:.2f}%\nhappy = {:.2f}%\nsadness = {:.2f}%\nsurprise = {:.2f}%".format(*tab[0]*100))
