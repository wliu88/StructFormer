#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from structformer.models.point_transformer import PointTransformerEncoderSmall


class FocalLoss(nn.Module):
    "Focal Loss"

    def __init__(self, gamma=2, alpha=.25):
        super(FocalLoss, self).__init__()
        # self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        # targets = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        # F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()


class EncoderMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, pt_dim=3, uses_pt=True):
        super(EncoderMLP, self).__init__()
        self.uses_pt = uses_pt
        self.output = out_dim
        d5 = int(in_dim)
        d6 = int(2 * self.output)
        d7 = self.output
        self.encode_position = nn.Sequential(
                nn.Linear(pt_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                )
        d5 = 2 * in_dim if self.uses_pt else in_dim
        self.fc_block = nn.Sequential(
            nn.Linear(int(d5), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(int(d6), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(d6, d7))

    def forward(self, x, pt=None):
        if self.uses_pt:
            if pt is None: raise RuntimeError('did not provide pt')
            y = self.encode_position(pt)
            x = torch.cat([x, y], dim=-1)
        return self.fc_block(x)


class RearrangeObjectsPredictorPCT(torch.nn.Module):

    def __init__(self, vocab_size,
                 num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu", encoder_num_layers=8,
                 use_focal_loss=False, focal_loss_gamma=2):
        super(RearrangeObjectsPredictorPCT, self).__init__()

        print("Object Selection Network with Point Transformer")

        # object encode will have dim 256
        self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 240 (point cloud) + 8 (position idx) + 8 (token type)
        self.mlp = EncoderMLP(256, 240, uses_pt=False)

        self.word_embeddings = torch.nn.Embedding(vocab_size, 240, padding_idx=0)
        self.token_type_embeddings = torch.nn.Embedding(2, 8)
        self.position_embeddings = torch.nn.Embedding(11, 8)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.rearrange_object_fier = nn.Sequential(nn.Linear(256, 256),
                                                   nn.LayerNorm(256),
                                                   nn.ReLU(),
                                                   nn.Linear(256, 128),
                                                   nn.LayerNorm(128),
                                                   nn.ReLU(),
                                                   nn.Linear(128, 1))

        ###########################
        if use_focal_loss:
            print("use focal loss")
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, xyzs, rgbs, object_pad_mask, sentence, sentence_pad_mask, token_type_index, position_index):

        batch_size = object_pad_mask.shape[0]
        num_objects = object_pad_mask.shape[1]

        #########################
        center_xyz, x = self.object_encoder(xyzs, rgbs)
        x = self.mlp(x, center_xyz)
        x = x.reshape(batch_size, num_objects, -1)

        #########################
        sentence = self.word_embeddings(sentence)

        #########################
        position_embed = self.position_embeddings(position_index)
        token_type_embed = self.token_type_embeddings(token_type_index)
        pad_mask = torch.cat([sentence_pad_mask, object_pad_mask], dim=1)

        sequence_encode = torch.cat([sentence, x], dim=1)
        sequence_encode = torch.cat([sequence_encode, position_embed, token_type_embed], dim=-1)
        #########################
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        sequence_encode = sequence_encode.transpose(1, 0)

        # convert to bool
        pad_mask = (pad_mask == 1)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(sequence_encode, src_key_padding_mask=pad_mask)
        encode = encode.transpose(1, 0)
        #########################
        obj_encodes = encode[:, -num_objects:, :]
        obj_encodes = obj_encodes.reshape(-1, obj_encodes.shape[-1])

        rearrange_obj_labels = self.rearrange_object_fier(obj_encodes).squeeze(dim=1)  # batch_size * num_objects

        predictions = {"rearrange_obj_labels": rearrange_obj_labels}

        return predictions

    def criterion(self, predictions, labels):

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            mask = gts == -100
            preds = preds[~mask]
            gts = gts[~mask]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "rearrange_obj_labels":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions