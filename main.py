# python libraries
import argparse
import time
import warnings

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# local files
# from model.simclr_model import *


# Define command-line arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Train a good baseline model on Noisy CIFAR-10/100 dataset')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose the dataset')
parser.add_argument('--noise_mode', type=str, default='clean_label',
                    choices=['clean_label', 'aggre_label', 'rand_1_label', 'rand_2_label', 'rand_3_label', 'worst_label', 'symmetric_flip_label', 'promix_100_label',
                             'promix_400_label'],
                    help='Noise mode for labels: random or human')
parser.add_argument('--symmetric_flip_prob', type=float, default=None, help='Probability of symmetric label flipping')
parser.add_argument('--feature_type', type=str, default='original',
                    choices=['original', 'transfer_learning', 'contrastive_learning', 'foundation_model'],
                    help='feature type for training linear model')
parser.add_argument('--encoder_name', type=str, default=None, choices=[None, 'resnet18', 'resnet34', 'resnet50', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])

# parser.add_argument('--num_epochs', type=int, default=20,
#                     help='Number of epochs/iterations for training')
parser.add_argument("--batch_size", default=64, type=int, help="Batch size used during feature extraction.")

# add parser: optimizer - sgd or adam or sgd with lr scheduler
# parser.add_argument('--optimizer', type=str, default='lbfgs', choices=['lbfgs'])

args = parser.parse_args()



def main():

    warnings.filterwarnings('ignore') # turn off the warnings (especially for sklearn convergence)

    print()
    print(f"================ {args.dataset} with {args.noise_mode} (symmetric flip prob = {args.symmetric_flip_prob}) ========================")
    # print()
    print(f"Linear model + feature by {args.feature_type}, encoder: {args.encoder_name}")
    # print(f"num_epochs: {args.num_epochs}, optimizer: {args.optimizer}, batch_size: {args.batch_size}")
    print()

    if args.feature_type == 'original' or args.feature_type == 'contrastive_learning': # in these cases, not need to upscale cifar images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.feature_type == 'transfer_learning' or args.feature_type == 'foundation_model': # need to upscale, cuz pre-trained on ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    # Load CIFAR-10 dataset with noisy labels
    dataset_path = './data'

    # Load CIFAR-10 dataset
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform)

    # Check if CUDA is available and set PyTorch to use GPU or CPU
    # Move model to GPU if available
    try:
        if torch.cuda.is_available():
            print()
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except AttributeError:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print("Using device:", device)
    print()

    start_time = time.time()
    # extract the features and get the training and testing data
    if args.feature_type == 'original': # directly get all the data
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

        # Extract the data and labels
        X_train, _ = next(iter(trainloader))    
        X_test, y_test = next(iter(testloader))

        X_train = X_train.view(X_train.size(0), -1).numpy()
        X_test = X_test.view(X_test.size(0), -1).numpy()
        y_test = y_test.numpy()
    else: # extract the features in a batch manner (otherwise, may run out of memory)

        # Load the  pre-trained feature extractor
        if args.feature_type == 'foundation_model':
            feature_extractor = torch.hub.load('facebookresearch/dinov2', args.encoder_name)
        elif args.feature_type == 'transfer_learning':
            if args.encoder_name == 'resnet18':
                feature_extractor = models.resnet18(pretrained=True)
            elif args.encoder_name == 'resnet34':
                feature_extractor = models.resnet34(pretrained=True)
            elif args.encoder_name == 'resnet50':
                feature_extractor = models.resnet50(pretrained=True)
            feature_extractor.fc = nn.Identity() # Replace the classification layer with an identity function
        elif args.feature_type == 'contrastive_learning':
            pretrained_model = torch.load(f'trained model/ckpt_{args.dataset}_{args.encoder_name}.pth')
            sd = {}
            for ke in pretrained_model['model']:
                nk = ke.replace('module.', '')
                sd[nk] = pretrained_model['model'][ke]
            feature_extractor = Encoder_cl(name=args.encoder_name)
            feature_extractor.load_state_dict(sd, strict=False)

        feature_extractor.to(device)
        feature_extractor.eval()

        # Extract the features
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

        X_train = []
        X_test, y_test = [], []

        # Extract features from training data (training labels are noisy, load later)
        for i, (inputs, _) in enumerate(trainloader):
            inputs = inputs.to(device)
            with torch.no_grad():
                features = feature_extractor(inputs)
                features = torch.flatten(features, start_dim=1).cpu().numpy()  # Flatten the features
            X_train.append(features)
            if (i+1) % 50 == 0:  # Check if (i+1) is divisible by 10
                print(f'Batch {i+1}/{len(trainloader)} of train data processed.')

        # Extract features and labels from test data
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            with torch.no_grad():
                features = feature_extractor(inputs)
                features = torch.flatten(features, start_dim=1).cpu().numpy()  # Flatten the features
            X_test.append(features)
            y_test.append(labels.numpy())
            if (i+1) % 50 == 0:  # Check if (i+1) is divisible by 10
                print(f'Batch {i+1}/{len(testloader)} of test data processed.')

        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)


    # load noisy training labels
    if args.dataset == 'cifar10':
        noise_file_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        noise_file_path = './data/CIFAR-100_human.pt'

    noise_file = torch.load(noise_file_path)

    if args.noise_mode == 'clean_label':
        y_train = noise_file['clean_label']
    elif args.noise_mode == 'aggre_label':
        y_train = noise_file['aggre_label']
    elif args.noise_mode == 'rand_1_label':
        y_train = noise_file['random_label1']
    elif args.noise_mode == 'rand_2_label':
        y_train = noise_file['random_label2']
    elif args.noise_mode == 'rand_3_label':
        y_train = noise_file['random_label3']
    elif args.noise_mode == 'worst_label':
        if args.dataset == 'cifar10':
            y_train = noise_file['worse_label']
        elif args.dataset == 'cifar100':
            y_train = noise_file['noisy_label']
    elif args.noise_mode == 'symmetric_flip_label':
        y_train = torch.load(f'{dataset_path}/CIFAR-10_symmetric_{args.symmetric_flip_prob}.pt')


    # Cross validation Define the parameter grid
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs'],
        'max_iter': [10, 20, 50, 100]
    }

    # Create a logistic regression model
    model = LogisticRegression(multi_class='multinomial')

    # Create the grid search object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print()
    print("Best parameters based on cross validation on noisy data: ", grid_search.best_params_)
    print()

    # Evaluate the model
    y_pred = grid_search.predict(X_test)
    print(f"Testing Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
    print()

    end_time = time.time()
    print('Total training time: {:.2f} seconds'.format(end_time - start_time))

if __name__ == '__main__':
    main()