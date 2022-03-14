import numpy as np
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Dataset import Dataset
from RedditNN import RedditNN


def train(args, model, dataset, optimizer):
    """
    Main training loop for RedditNN models
    :param args: (ArgumentParser) arguments specified when running program
    :param model: (RedditNN) neural network model
    :param dataset: (Dataset) class containing data for training/validation/testing
    :return: modifies 'model' in place
    """

    model.train()

    dataloader = DataLoader(dataset, batch_size=args.bs)

    lossFCN = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for e in range(args.epochs):

        # re-initialize model state at start of each Epoch
        h, c = model.init_state(dataset.sequenceLength)

        for b, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()

            predictedY, (h, c) = model(x, (h, c))
            #print(y.size())
            #print(predictedY.size())
            loss = lossFCN(predictedY.transpose(1, 2), y)

            h, c = h.detach(), c.detach()

            loss.backward()
            optimizer.step()

            if b % 100 == 0:
                print({'epoch': e, 'batch': b, 'loss': loss.item()})


def predict(args, model, dataset):
    """
    Used to query a prediction from a trained RedditNN model
    :param args: (ArgumentParser) arguments specified when running program
    :param model: (RedditNN) neural network model
    :param dataset: (Dataset) class containing data for training/validation/testing
    :return: query (List[Str]) sentence generated by the RedditNN model
    """

    model.eval()

    query = args.query.split()
    h, c = model.init_state(len(query))

    for i in range(args.prediction_length):
        x = torch.tensor([[dataset.wordToIdx[q] for q in query[i:]]])

        with torch.no_grad():
            y_pred, (h, c) = model(x, (h, c))

        #print(y_pred.size())
        #print(len(dataset.uniqueWords))

        logits = y_pred[0][-1]
        probs = nn.functional.softmax(logits, dim=0).detach().numpy()
        idx = np.random.choice(len(logits), p=probs)
        query.append(dataset.idxToWord[idx])

    return query


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--query", type=str, default="does anyone else")
    parser.add_argument("--prediction_length", type=int, default=50)
    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--use_checkpoint", default=False, action='store_true')
    parser.add_argument("--model_name", type=str, default="model1.pth")
    parser.add_argument("--optim_name", type=str, default="optim1.pth")
    args = parser.parse_args()

    # load in Dataset to train NN
    dataset = Dataset(args)
    #dataset.scrapeNewPostsToFile('CasualConversation', numPosts=100, appendFile=True)
    dataset.loadPostsFromFile()

    # initialize a new RedditNN model
    model = RedditNN(dataset)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train the new RedditNN model
    if args.train:
        if args.use_checkpoint:
            print("Continuing training from checkpoint ", args.model_name, "\n")
            model.load_state_dict(torch.load(args.model_name))
            optimizer.load_state_dict(torch.load(args.optim_name))

            for n in model.state_dict():
                print(n)
        else:
            print("Training new model from scratch...\n")
'''
        # launch main training loop
        train(args, model, dataset, optimizer)

        # save the model
        torch.save(model.state_dict(), args.model_name)
        torch.save(optimizer.state_dict(), args.optim_name)
        print("Saved PyTorch Model State to ", args.model_name, "\n")
        print(model.state_dict())

    # if training is not desired, simply load the model
    else:
        print("Using pre-existing model ", args.model_name)
        model.load_state_dict(torch.load(args.model_name))

    # query predictions from trained RedditNN model
    result = predict(args, model, dataset)

    resultStr = ""
    for w in result:
        resultStr += " " + w

    print(resultStr)
'''