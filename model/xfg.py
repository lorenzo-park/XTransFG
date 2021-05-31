from scipy import ndimage

import copy
import torch
import torch.nn as nn
import numpy as np

from model.vit import FFN, Transformer, np2th, Block


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """
        Args:
            input_feat: (N, L, D)
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class XBlock(nn.Module):
    def __init__(self, config):
        super(XBlock, self).__init__()
        self.attn = nn.MultiheadAttention(config.hidden_size, config.transformer.num_heads)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn = FFN(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, q, k, v):
        h = q
        x, weights = self.attn(q, k, v)
        x = self.attention_norm(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights


class XFGCrossAttn(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGCrossAttn, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)

        self.cross_layers = XBlock(config)

        self.transformer = Transformer(config)
        self.txt_token_proj = nn.Linear(80, 325)
        self.head = nn.Linear(config.hidden_size, num_classes)

        self.img_pos_embedding = TrainablePositionalEncoding(325, config.hidden_size, dropout=config.dropout)
        self.txt_pos_embedding = TrainablePositionalEncoding(325, config.hidden_size, dropout=config.dropout)

    def forward(self, img, txt_tokens):
        img_tokens, _ = self.transformer(img)

        txt_tokens = txt_tokens.permute(0, 2, 1)
        txt_tokens = self.txt_token_proj(txt_tokens)
        txt_tokens = txt_tokens.permute(0, 2, 1)

        img_tokens = self.img_pos_embedding(img_tokens)
        txt_tokens = self.txt_pos_embedding(txt_tokens)

        img_tokens, _ = self.cross_layers(img_tokens, txt_tokens, txt_tokens)

        logits = self.head(img_tokens[:, 0])
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


class EncoderConcat(nn.Module):
    def __init__(self, config):
        super(EncoderConcat, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer.num_layers_fusion):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights



class XFGConcat(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGConcat, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)
        self.transformer = Transformer(config)

        self.encoder = EncoderConcat(config)

        self.head = nn.Linear(config.hidden_size, num_classes)

        self.pos_embedding = TrainablePositionalEncoding(80+325, config.hidden_size, dropout=config.dropout)

        if config.transformer.shared_cls:
            self.cls_token = self.transformer.cls_token
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img, txt_tokens):
        cls_token = self.cls_token.expand(img.shape[0], -1, -1)

        img_tokens, _ = self.transformer(img)
        tokens = torch.cat([txt_tokens, img_tokens], dim=1)
        tokens = self.pos_embedding(tokens) + tokens
        tokens = torch.cat([cls_token, tokens], dim=1)

        tokens, _ = self.encoder(tokens)

        logits = self.head(tokens[:, 0])
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)