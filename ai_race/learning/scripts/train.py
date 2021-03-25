'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import sys
import io
import argparse
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torchsummary import summary
from mobilenetv2 import MobileNetV2
from MyDataSet import MyDataset
from samplenet import SampleNet, SimpleNet, SimpleNet2
import vitnet

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../config")
import learning_config

DISCRETIZATION = learning_config.Discretization_number


def main(trial):
    print('optuna trial={}'.format(trial.number))
    # Parse arguments.
    args = parse_args()
    
    # Set device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare dataset.
    np.random.seed(seed=0)
    image_dataframe = pd.read_csv(args.data_csv, engine='python', header=None)
    image_dataframe = image_dataframe.reindex(np.random.permutation(image_dataframe.index))
    test_num = int(len(image_dataframe) * 0.2)
    train_dataframe = image_dataframe[test_num:]
    test_dataframe = image_dataframe[:test_num]
    train_data = MyDataset(train_dataframe, transform=transforms.ToTensor())
    test_data = MyDataset(test_dataframe, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    
    print('data set')
    # Set a model.
    if args.model == 'resnet18':
        model = models.resnet18()
        model.fc = torch.nn.Linear(512, DISCRETIZATION)
    elif args.model == 'mobilenetv2':
        model = MobileNetV2(num_classes=3)
    elif args.model == 'samplenet':
        model = SampleNet(DISCRETIZATION)
    elif args.model == 'simplenet':
        model = SimpleNet(DISCRETIZATION, init_maxpool=2, use_gap=False)
    elif args.model == 'simplenet2':
        model = SimpleNet2()
    elif args.model == 'vit2':
        model = vitnet.ViT2Net()
    elif args.model == 'vit2s':
        model = vitnet.ViT2NetS()
    elif args.model == 'vit2m':
        model = vitnet.ViT2NetM()
    else:
        raise NotImplementedError()

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 320, 240), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    model.train()
    model = model.to(device)
    summary(model, (3, 320, 240))

    print('model set')
    # Set loss function and optimization function.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    print('optimizer set')
    
    # Train and test.
    print('Train starts')
    test_acc_best = 0
    test_acc_best_epoch = 0
    last_model_ckpt_path = None
    for epoch in range(args.n_epoch):
        # Train and test a model.
        train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
        
        # Output score.
        if(epoch%args.test_interval == 0):
            test_acc, test_loss = test(model, device, test_loader, criterion)

            stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
            print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))
            if test_acc > test_acc_best:
                test_acc_best = test_acc
                test_acc_best_epoch = epoch + 1
                model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, trial.number, epoch+1)
                torch.save(model.state_dict(), model_ckpt_path)
                print('Saved a model checkpoint at {}'.format(model_ckpt_path))
                if last_model_ckpt_path is not None:
                    os.remove(last_model_ckpt_path)
                last_model_ckpt_path = model_ckpt_path
            else:
                if (epoch + 1) - test_acc_best_epoch >= args.early_stopping:
                    break

        else:    
            stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}' #, test acc: {:<8}, test loss: {:<8}'
            print(stdout_temp.format(epoch+1, train_acc, train_loss)) #, test_acc, test_loss))
    msg = 'trial={} test_acc_best={} test_acc_best_epoch={}'.format(trial.number, test_acc_best, test_acc_best_epoch)
    print(msg)
    with open(os.path.join(args.model_ckpt_dir, 'optuna.log'), 'a') as f:
        f.write(msg + '\n')

    return test_acc_best


def train(model, device, train_loader, criterion, optimizer):
    model.train()

    output_list = []
    target_list = []
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward processing.
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward processing.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Set data to calculate score.
        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in targets]
        running_loss += loss.item()

        # Calculate score at present.
        train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
        if batch_idx % 10 == 0 and batch_idx != 0:
            stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
            print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))

    # Calculate score.
    train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

    return train_acc, train_loss


def test(model, device, test_loader, criterion):
    model.eval()

    output_list = []
    target_list = []
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # Forward processing.
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Set data to calculate score.
        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in targets]
        running_loss += loss.item()
        
    test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

    print('confusion_matrix')
    print(confusion_matrix(output_list, target_list))
    print('classification_report')
    print(classification_report(output_list, target_list))

    return test_acc, test_loss



def calc_score(output_list, target_list, running_loss, data_loader):
    # Calculate accuracy.
    #result = classification_report(output_list, target_list) #, output_dict=True)
    #acc = round(result['weighted avg']['f1-score'], 6)
    acc = round(f1_score(output_list, target_list, average='micro'), 6)
    loss = round(running_loss / len(data_loader.dataset), 6)

    return acc, loss


def parse_args():
    # Set arguments.
    arg_parser = argparse.ArgumentParser(description="Image Classification")
    
    arg_parser.add_argument("--batch_size", type=int, default=20)
    arg_parser.add_argument("--dataset_name", type=str, default='sim_race')
    arg_parser.add_argument("--data_csv", type=str, default=os.environ['HOME'] + '/Images_from_rosbag/_2020-11-05-01-45-29_2/_2020-11-05-01-45-29.csv')
    arg_parser.add_argument("--early_stopping", type=int, default=3)
    arg_parser.add_argument("--model", type=str, default='resnet18')
    arg_parser.add_argument("--model_name", type=str, default='joycon_ResNet18')
    arg_parser.add_argument("--model_ckpt_dir", type=str, default=os.environ['HOME'] + '/work/experiments/models/checkpoints/')
    arg_parser.add_argument("--model_ckpt_path_temp", type=str, default=os.environ['HOME'] + '/work/experiments/models/checkpoints/{}_{}_epoch={}.pth')
    arg_parser.add_argument('--n_epoch', default=20, type=int, help='The number of epoch')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    arg_parser.add_argument('--test_interval', default=5, type=int, help='test interval')
    arg_parser.add_argument('--save_model_interval', default=5, type=int, help='save model interval')

    args = arg_parser.parse_args()

    # Make directory.
    os.makedirs(args.model_ckpt_dir, exist_ok=True)

    print(args.data_csv)
    # Validate paths.
    assert os.path.exists(args.data_csv)
    assert os.path.exists(args.model_ckpt_dir)

    return args


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(main, n_trials=500)
    print('best_params={}'.format(study.best_params))
    print("finished successfully.")
    os._exit(0)
