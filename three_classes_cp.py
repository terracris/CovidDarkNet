import os
import shutil
import random
from pathlib import Path
import numpy as np
from fastai.vision.all import *
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Functions for moving and organizing data
def create_directories(root_dir, directories):
    for directory in directories:
        os.mkdir(os.path.join(root_dir, directory))

def rename_directories(root_dir, source_dirs, class_names):
    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

def create_class_directories(root_dir, directories, class_names):
    for directory in directories:
        for c in class_names:
            os.mkdir(os.path.join(root_dir, directory, c))

def move_images(root_dir, class_names, directory, percentage):
    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        sample_size = int(len(images) * percentage)
        selected_images = random.sample(images, sample_size)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, directory, c, image)
            shutil.move(source_path, target_path)

# DarkCovidNet (Modified Darknet Model)
def conv_block(ni, nf, size=3, stride=1):
    for_pad = lambda s: s if s > 2 else 3
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=size, stride=stride,
                  padding=(for_pad(size) - 1)//2, bias=False), 
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)  
    )

def triple_conv(ni, nf):
    return nn.Sequential(
        conv_block(ni, nf),
        conv_block(nf, ni, size=1),  
        conv_block(ni, nf)
    )

def maxpooling():
    return nn.MaxPool2d(2, stride=2)
# Model definition
model = nn.Sequential(
    conv_block(3, 8),
    maxpooling(),
    conv_block(8, 16),
    maxpooling(),
    triple_conv(16, 32),
    maxpooling(),
    triple_conv(32, 64),
    maxpooling(),
    triple_conv(64, 128),
    maxpooling(),
    triple_conv(128, 256),
    conv_block(256, 128, size=1),
    conv_block(128, 256),
    nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
    Flatten(),
    nn.Linear(507, 3)
)

def main():

    # Configuration
    root_dir = 'COVID-19 Radiography Database'
    class_names = ['normal', 'viral', 'covid']
    source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
    train_percentage = 0.70 # 70% of the data will be used for training
    test_percentage = 0.50  # 50% of the remaining data (15%) will be used for testing
    validation_percentage = 1 # 100% of the remaining data (15%) will be used for validation
    directories = ['test', 'train', 'valid']

    # Organize the data
    if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
        create_directories(root_dir, directories)
        rename_directories(root_dir, source_dirs, class_names)
        create_class_directories(root_dir, directories, class_names)

        move_images(root_dir, class_names, 'train', train_percentage)
        move_images(root_dir, class_names, 'valid', validation_percentage)
        move_images(root_dir, class_names, 'test', test_percentage)

    # Data loading and preparation
    path = Path(root_dir)

    # DataBlock and DataLoader creation
    np.random.seed(41)
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
                    get_y=parent_label,
                    item_tfms=Resize(256),
                    batch_tfms=aug_transforms(size=256, min_scale=0.75))
    dls = dblock.dataloaders(path)

    # Model training and evaluation
    learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
    learn.fit_one_cycle(100, lr_max=3e-3)
    learn.save('covid_model')

    # Model evaluation and metrics calculation
    predictions, targets = learn.get_preds(ds_idx=1) # predictions on validation set

    predictions = np.argmax(predictions, axis=1)
    correct = 0
    for idx, pred in enumerate(predictions):
        if pred == targets[idx]:
            correct += 1
    accuracy = correct / len(predictions)
    print(len(predictions), correct, accuracy)

    np.set_printoptions(threshold=np.inf) # shows whole confusion matrix
    cm1 = confusion_matrix(targets, predictions)
    print(cm1)

    y_true1 = targets
    y_pred1 = predictions
    target_names = ['Covid-19', 'No_findings', 'Pneumonia']
    print(classification_report(y_true1, y_pred1, target_names=target_names))

    interp = ClassificationInterpretation.from_learner(learn)

    # Assuming `interp` is your ClassificationInterpretation object
    _, targs, decoded = interp.learn.get_preds(dl=interp.dl, with_decoded=True)

    # Convert multi-dimensional array to single-label array
    decoded_single_label = torch.argmax(decoded, dim=1).numpy()

    # Calculate confusion matrix
    cm = confusion_matrix(targs.numpy(), decoded_single_label)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm1, interpolation='nearest', cmap=plt.cm1.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()

    # Add class labels
    # Add class labels
    num_classes = len(interp.dl.vocab)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, interp.dl.vocab)
    plt.yticks(tick_marks, interp.dl.vocab)

    # Add annotations
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Save the figure
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    main()