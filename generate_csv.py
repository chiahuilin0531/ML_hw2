import os
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import glob

def generate_train_csv():
    with open('./dataset/train_img.csv', 'w') as img_file:
        with open('./dataset/train_label.csv', 'w') as label_file:
            with open('./dataset/character_label.csv', 'w') as character_file:
                img_writer = csv.writer(img_file)
                label_writer = csv.writer(label_file)
                character_writer = csv.writer(character_file)
                character_writer.writerow(["index", "character name"])
                characters = glob.glob(os.path.join("dataset", "theSimpsons-train", "train","*"))
                print()
                print("Found", len(characters), "characters.")
                print()
                for i, c in enumerate(characters):
                    character_writer.writerow([i,c.split(os.sep)[-1]])
                    imgs = glob.glob(os.path.join(c, '*'))
                    print("\tFound", len(imgs), "images of", c.split(os.sep)[-1])
                    for img in imgs:
                        img_writer.writerow([img.replace(os.sep, '/')])
                        label_writer.writerow([i])
                label_file.close()
                img_file.close()
                character_file.close()

def generate_test_csv():
    with open('./dataset/test_img.csv', 'w') as file:
        writer = csv.writer(file)
        # for img in glob.glob(os.path.join("dataset", "theSimpsons-test", "test","*")):
            # writer.writerow([img.replace(os.sep, '/')])
        for img in range(10791):
            writer.writerow([os.path.join("dataset", "theSimpsons-test", "test", str(img+1)+".jpg").replace(os.sep, '/')])
        file.close()

if __name__ == '__main__':
    # generate_train_csv()
    generate_test_csv()
    # img = pd.read_csv('./dataset/train_img.csv', header=None)
    # label = pd.read_csv('./dataset/train_label.csv', header=None)
    # print(img)
    # print(label)