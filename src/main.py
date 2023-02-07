import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.models import *
from src.dataset import *
from config.constants import *

BATCH_SIZE = 32
BASE_DIR = Path(__file__).parents[1]
LEARNING_RATE = 0.001

ALL_MODELS = {
    'invariantmlp': InvariantMLP(),
}

def create_dataloaders(
        dataset: any,
        batch_size: int = 32,
        split_ratios: list = [0.8, 0.2],
        random_seed: int = 42,
):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    assert sum(split_ratios) == 1
    split_idx = [int(np.floor(i * dataset_size)) for i in split_ratios]
    split_idx.insert(0, 0)
    split_idx = [sum(split_idx[:i+1]) for i in range(len(split_idx))]
    split_idx[-1] = -1

    split_idx = [indices[split_idx[i]:split_idx[i + 1]] for i in range(len(split_idx) - 1)]
    samplers = [SubsetRandomSampler(x) for x in split_idx]
    dataloaders = [DataLoader(dataset, batch_size=batch_size, sampler=x) for x in samplers]
    print('Generated %s dataloaders' % len(dataloaders))
    return dataloaders

def run_epoch(
        dataloader: any = None,
        dataset: str = 'original',
        model: any = None,
        model_name: str = 'invariantmlp',
        model_file: str = 'invariantmlp_original',
        train: bool = False,
        loss_fn: any = None,
        optimizer: any = None,
        print_result: bool = False,
):
    if train:
        assert optimizer is not None

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    if model is None:
        model_name = model_name.lower().strip()
        model_file = model_file.lower().strip()
        assert model_name in ALL_MODELS.keys()
        model = ALL_MODELS[model_name]
        saved_model_path = BASE_DIR.joinpath('data/saved_models/' + model_file + '.pt')
        model.load_state_dict(torch.load(saved_model_path)['model_state_dict'])
        if not train:
            model.eval()

    if dataloader is None:
        dataset = dataset.lower().strip()
        assert dataset in ALL_DATASETS
        dataset = KreuzerSkarkeDataset(load_projections=True, projections_file=dataset)
        dataloader = create_dataloaders(dataset, split_ratios=[0.8, 0.2], batch_size=BATCH_SIZE)[int(not train)]

    correct_pred, losses, num_samples, batch_counter = 0, 0, 0, 0

    for (X, y) in dataloader:
        if train:
            optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        pred = np.argmax(pred.detach().numpy(), axis=1)
        y = y.detach().numpy()
        correct_pred += (pred == y).sum()
        losses += loss.item()
        num_samples += len(X)
        batch_counter += 1

        if train:
            loss.backward()
            optimizer.step()

    epoch_loss, epoch_acc = losses / batch_counter, correct_pred / num_samples
    if print_result:
        print('Epoch Loss: %.2f\t Epoch Accuracy: %.2f' % (epoch_loss, epoch_acc))
    return losses / batch_counter, correct_pred / num_samples

def train(
        dataset: str,
        model: str,
        output_tag: str = None,
        num_epochs: int = 20,
):
    dataset = dataset.lower().strip()
    assert dataset in ALL_DATASETS

    model = model.lower().strip()
    assert model in ALL_MODELS.keys()

    if output_tag is None:
        output_tag = model + '_' + dataset
    summary_writer_path = BASE_DIR.joinpath('data/runs/tensorboard')
    saved_model_path = BASE_DIR.joinpath('data/saved_models/' + output_tag + '.pt')
    saved_results_path = BASE_DIR.joinpath('data/saved_results/' + output_tag + '.csv')

    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file=dataset)
    train_dataloader, val_dataloader = create_dataloaders(dataset, split_ratios=[0.8, 0.2], batch_size=BATCH_SIZE)
    model = ALL_MODELS[model]
    writer = SummaryWriter(summary_writer_path)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    max_acc = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = run_epoch(
            dataloader=train_dataloader,
            model=model,
            train=True,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            dataloader=val_dataloader,
            model=model,
            loss_fn=loss_fn,
        )

        print('Epoch: %d\n\t Training Loss: %.2f\t Training Accuracy: %.2f' % (epoch, train_loss, train_acc ))
        print('\t Validation Loss: %.2f\t Validation Accuracy: %.2f' % (val_loss, val_acc))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        writer.add_scalars('loss/train', {output_tag: train_loss}, epoch)
        writer.add_scalars('accuracy/train', {output_tag: train_acc}, epoch)
        writer.add_scalars('loss/val', {output_tag: val_loss}, epoch)
        writer.add_scalars('accuracy/val', {output_tag: val_acc}, epoch)
        writer.flush()

        if max_acc <= val_acc:
            max_acc = val_acc
            print('Saving model checkpoint to %s for epoch %d' % (saved_model_path, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
            }, saved_model_path)

    results_df = pd.DataFrame({
        'epoch': list(range(num_epochs)),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    })

    print('Saving results as a csv in  %s' % saved_results_path)
    results_df.to_csv(saved_results_path)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='original', type=str, help='Dataset type. Should be from this list: ' + ', '.join(ALL_DATASETS))
    parser.add_argument('--model', default='invariantmlp', type=str, help='Model Type. Should be from this list: ' + ', '.join(ALL_MODELS.keys()))
    parser.add_argument('--output_tag', default=None, type=str, help='Output tag used to save results or fetch saved results. Required if "--eval" flag is used')
    parser.add_argument('--eval', action='store_true', help='Specify if the script needs to be run in eval mode')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs for training')
    args = vars(parser.parse_args())

    if not args['eval']:
        train(args['dataset'], args['model'], output_tag=args['output_tag'], num_epochs=args['num_epochs'])
    else:
        run_epoch(dataset=args['dataset'], model_name=args['model'], model_file=args['output_tag'], print_result=True)