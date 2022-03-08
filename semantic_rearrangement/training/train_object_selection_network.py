from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
import time
import os
import tqdm

import argparse
from omegaconf import OmegaConf
from collections import defaultdict
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from semantic_rearrangement.data.object_set_refer_dataset import ObjectSetReferDataset
from semantic_rearrangement.models.object_selection_network import RearrangeObjectsPredictorPCT
from semantic_rearrangement.data.tokenizer import Tokenizer


def evaluate(gts, predictions, keys, debug=True, return_classification_dict=False):
    """
    :param gts: expect a list of tensors
    :param predictions: expect a list of tensor
    :return:
    """

    total_scores = 0
    for key in keys:
        predictions_for_key = torch.cat(predictions[key], dim=0)
        gts_for_key = torch.cat(gts[key], dim=0)

        predicted_classes = predictions_for_key > 0.5
        assert len(gts_for_key) == len(predicted_classes)

        target_indices = gts_for_key != -100

        gts_for_key = gts_for_key[target_indices]
        predicted_classes = predicted_classes[target_indices]
        num_objects = len(predicted_classes)

        if debug:
            print(num_objects)
            print(gts_for_key.shape)
            print(predicted_classes.shape)
            print(target_indices.shape)
            print("Groundtruths:")
            print(gts_for_key[:100])
            print("Predictions")
            print(predicted_classes[:100])

        accuracy = torch.sum(gts_for_key == predicted_classes) / len(gts_for_key)
        print("{} objects -- {} accuracy: {}".format(num_objects, key, accuracy))
        total_scores += accuracy

        report = classification_report(gts_for_key.detach().cpu().numpy(), predicted_classes.detach().cpu().numpy(),
                                       output_dict=True)
        print(report)

    if not return_classification_dict:
        return total_scores
    else:
        return report


def train_model(cfg, model, data_iter, optimizer, warmup, num_epochs, device, save_best_model, grad_clipping=1.0):

    if save_best_model:
        best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
        print("best model will be saved to {}".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_score = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0
        gts = defaultdict(list)
        predictions = defaultdict(list)

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            for step, batch in enumerate(data_iter["train"]):
                optimizer.zero_grad()

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                rgbs = batch["rgbs"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["rearrange_obj_labels"]:
                    targets[key] = batch[key].to(device, non_blocking=True)

                preds = model.forward(xyzs, rgbs, object_pad_mask, sentence, sentence_pad_mask, token_type_index, position_index)
                loss = model.criterion(preds, targets)

                loss.backward()

                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                optimizer.step()
                epoch_loss += loss

                for key in ["rearrange_obj_labels"]:
                    gts[key].append(targets[key].detach())
                    predictions[key].append(preds[key].detach())

                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

        warmup.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))
        evaluate(gts, predictions, ["rearrange_obj_labels"])

        score = validate(model, data_iter["valid"], epoch, device)
        if save_best_model and score > best_score:
            print("Saving best model so far...")
            best_score = score
            save_model(best_model_dir, cfg, epoch, model)

    return model


def validate(model, data_iter, epoch, device):
    """
    helper function to evaluate the model

    :param model:
    :param data_iter:
    :param epoch:
    :param device:
    :return:
    """

    model.eval()

    epoch_loss = 0
    gts = defaultdict(list)
    predictions = defaultdict(list)
    with torch.no_grad():

        with tqdm.tqdm(total=len(data_iter)) as pbar:
            for step, batch in enumerate(data_iter):

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                rgbs = batch["rgbs"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["rearrange_obj_labels"]:
                    targets[key] = batch[key].to(device, non_blocking=True)

                preds = model.forward(xyzs, rgbs, object_pad_mask, sentence, sentence_pad_mask, token_type_index,
                                      position_index)
                loss = model.criterion(preds, targets)

                for key in ["rearrange_obj_labels"]:
                    gts[key].append(targets[key])
                    predictions[key].append(preds[key])

                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))

    score = evaluate(gts, predictions, ["rearrange_obj_labels"])
    return score


def infer_once(model, batch, device):
    """
    helper function to evaluate the model

    :param model:
    :param data_iter:
    :param epoch:
    :param device:
    :return:
    """

    model.eval()

    gts = defaultdict(list)
    predictions = defaultdict(list)
    with torch.no_grad():

        # input
        xyzs = batch["xyzs"].to(device, non_blocking=True)
        rgbs = batch["rgbs"].to(device, non_blocking=True)
        sentence = batch["sentence"].to(device, non_blocking=True)
        object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
        sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
        token_type_index = batch["token_type_index"].to(device, non_blocking=True)
        position_index = batch["position_index"].to(device, non_blocking=True)

        # output
        targets = {}
        for key in ["rearrange_obj_labels"]:
            targets[key] = batch[key].to(device, non_blocking=True)

        preds = model.forward(xyzs, rgbs, object_pad_mask, sentence, sentence_pad_mask, token_type_index,
                              position_index)

        for key in ["rearrange_obj_labels"]:
            gts[key].append(targets[key])
            predictions[key].append(preds[key])

    return gts, predictions


def save_model(model_dir, cfg, epoch, model, optimizer=None, scheduler=None):
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(model_dir, "model.tar"))
    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_model(model_dir, dirs_cfg):
    """
    Load transformer model
    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """
    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    if dirs_cfg:
        cfg = OmegaConf.merge(cfg, dirs_cfg)

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    # initialize model
    model_cfg = cfg.model
    model = RearrangeObjectsPredictorPCT(vocab_size,
                                         num_attention_heads=model_cfg.num_attention_heads,
                                         encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                         encoder_dropout=model_cfg.encoder_dropout,
                                         encoder_activation=model_cfg.encoder_activation,
                                         encoder_num_layers=model_cfg.encoder_num_layers,
                                         use_focal_loss=model_cfg.use_focal_loss,
                                         focal_loss_gamma=model_cfg.focal_loss_gamma)
    model.to(cfg.device)

    # load state dicts
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = None
    if "optimizer_state_dict" in checkpoint:
        training_cfg = cfg.training
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if "scheduler_state_dict" in checkpoint:
        scheduler = None
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    return cfg, tokenizer, model, optimizer, scheduler, epoch


def run_model(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = ObjectSetReferDataset(data_cfg.dirs, data_cfg.index_dirs, "train", tokenizer,
                                          data_cfg.max_num_all_objects,
                                          data_cfg.max_num_shape_parameters,
                                          data_cfg.max_num_rearrange_features,
                                          data_cfg.max_num_anchor_features,
                                          data_cfg.num_pts)

    valid_dataset = ObjectSetReferDataset(data_cfg.dirs, data_cfg.index_dirs, "valid", tokenizer,
                                          data_cfg.max_num_all_objects,
                                          data_cfg.max_num_shape_parameters,
                                          data_cfg.max_num_rearrange_features,
                                          data_cfg.max_num_anchor_features,
                                          data_cfg.num_pts)

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    num_workers=data_cfg.num_workers,
                                    collate_fn=ObjectSetReferDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    num_workers=data_cfg.num_workers,
                                    collate_fn=ObjectSetReferDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory)

    # load model
    model_cfg = cfg.model
    model = RearrangeObjectsPredictorPCT(vocab_size,
                                         num_attention_heads=model_cfg.num_attention_heads,
                                         encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                         encoder_dropout=model_cfg.encoder_dropout,
                                         encoder_activation=model_cfg.encoder_activation,
                                         encoder_num_layers=model_cfg.encoder_num_layers,
                                         use_focal_loss=model_cfg.use_focal_loss,
                                         focal_loss_gamma=model_cfg.focal_loss_gamma)
    model.to(cfg.device)

    training_cfg = cfg.training
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.l2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.lr_restart)
    warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=training_cfg.warmup,
                                    after_scheduler=scheduler)

    train_model(cfg, model, data_iter, optimizer, warmup, training_cfg.max_epochs, cfg.device, cfg.save_best_model)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "model")
        print("Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(model_dir, cfg, cfg.max_epochs, model, optimizer, scheduler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str)
    parser.add_argument("--main_config", help='config yaml file for the model',
                        default='../configs/object_selection_network.yaml',
                        type=str)
    parser.add_argument("--dirs_config", help='config yaml file for directories',
                        default='../configs/data/circle_dirs.yaml',
                        type=str)
    args = parser.parse_args()

    # # debug
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects"

    assert os.path.exists(args.main_config), "Cannot find config yaml file at {}".format(args.main_config)
    assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dir_config)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")

    main_cfg = OmegaConf.load(args.main_config)
    dirs_cfg = OmegaConf.load(args.dirs_config)
    cfg = OmegaConf.merge(main_cfg, dirs_cfg)
    cfg.dataset_base_dir = args.dataset_base_dir
    OmegaConf.resolve(cfg)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_model(cfg)