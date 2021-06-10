from scipy import ndimage

import copy
import math
import torch
import torch.nn as nn
import numpy as np

from model.vit import FFN, Transformer, np2th, Block, Attention


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


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


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer.num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer.num_layers):
            layer = XBlock(config)
            self.layer.append(copy.deepcopy(layer))
        self.decoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states_img, hidden_states_txt):
        attn_weights = []
        hidden_states = hidden_states_img
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states, hidden_states_txt, hidden_states_txt)
            attn_weights.append(weights)
        return self.decoder_norm(hidden_states), attn_weights


class XBlock(nn.Module):
    def __init__(self, config):
        super(XBlock, self).__init__()
        self.self_attn = Attention(config)
        self.self_attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.cross_attn = CrossAttention(config)
        self.cross_attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn = FFN(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, q, k, v):
        h = q

        x, weights = self.self_attn(q)
        x = self.self_attention_norm(x)
        x = x + h

        h = x
        x, weights = self.cross_attn(x, k, v)
        x = self.cross_attention_norm(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights


class XFGCrossAttnDR(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGCrossAttnDR, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)

        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)

        self.img_token_proj = nn.Linear(config.visual_token_len, config.proj_dim)
        self.txt_token_proj = nn.Linear(config.max_len, config.proj_dim+1)

        self.transformer = Transformer(config)
        # self.txt_token_proj = nn.Linear(config.max_len, config.max_len)
        self.head = nn.Linear(config.hidden_size, num_classes)

        self.img_pos_embedding = TrainablePositionalEncoding(config.proj_dim+1, config.hidden_size, dropout=config.dropout)
        self.txt_pos_embedding = TrainablePositionalEncoding(config.proj_dim+1, config.hidden_size, dropout=config.dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img, txt_tokens):
        img_tokens, _ = self.transformer(img)
        img_tokens = img_tokens.permute(0, 2, 1)
        img_tokens = self.img_token_proj(img_tokens)
        img_tokens = img_tokens.permute(0, 2, 1)

        cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        img_tokens = torch.cat([cls_token, img_tokens], dim=1)

        txt_tokens = txt_tokens.permute(0, 2, 1)
        txt_tokens = self.txt_token_proj(txt_tokens)
        txt_tokens = txt_tokens.permute(0, 2, 1)

        img_tokens = self.img_pos_embedding(img_tokens)
        txt_tokens = self.txt_pos_embedding(txt_tokens)

        txt_tokens, _ = self.encoder(txt_tokens)
        img_tokens, attn_weights = self.decoder(img_tokens, txt_tokens)

        logits = self.head(img_tokens[:, 0])
        return logits, attn_weights

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


class XFGNoCrossAttnDR(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGNoCrossAttnDR, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)

        self.encoder = Encoder(config.encoder)
        # self.decoder = Decoder(config.decoder)

        self.img_token_proj = nn.Linear(config.visual_token_len, config.proj_dim)

        self.transformer = Transformer(config)
        # self.txt_token_proj = nn.Linear(config.max_len, config.max_len)
        self.head = nn.Linear(config.hidden_size, num_classes)

        self.img_pos_embedding = TrainablePositionalEncoding(config.proj_dim+1, config.hidden_size, dropout=config.dropout)
        # self.txt_pos_embedding = TrainablePositionalEncoding(config.max_len, config.hidden_size, dropout=config.dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img, txt_tokens):
        img_tokens, _ = self.transformer(img)
        img_tokens = img_tokens.permute(0, 2, 1)
        img_tokens = self.img_token_proj(img_tokens)
        img_tokens = img_tokens.permute(0, 2, 1)

        # txt_tokens = txt_tokens.permute(0, 2, 1)
        # txt_tokens = self.txt_token_proj(txt_tokens)
        # txt_tokens = txt_tokens.permute(0, 2, 1)

        cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        img_tokens = torch.cat([cls_token, img_tokens], dim=1)

        img_tokens = self.img_pos_embedding(img_tokens)
        # txt_tokens = self.txt_pos_embedding(txt_tokens)

        img_tokens, attn_weights = self.encoder(img_tokens)
        # img_tokens, attn_weights = self.decoder(img_tokens, txt_tokens)

        logits = self.head(img_tokens[:, 0])
        return logits, attn_weights

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


class XFGCrossAttnRec(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGCrossAttnRec, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)

        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)

        # self.rec_encoder = Encoder(config.rec_encoder)

        self.img_token_proj = nn.Linear(config.visual_token_len, config.proj_dim)
        self.txt_token_proj = nn.Linear(config.max_len, config.proj_dim+1)
        self.txt_token_proj_inv = nn.Linear(config.proj_dim, config.max_len)

        self.transformer = Transformer(config)
        # self.txt_token_proj = nn.Linear(config.max_len, config.max_len)
        self.head = nn.Linear(config.hidden_size, num_classes)

        self.img_pos_embedding = TrainablePositionalEncoding(config.proj_dim+1, config.hidden_size, dropout=config.dropout)
        self.txt_pos_embedding = TrainablePositionalEncoding(config.proj_dim+1, config.hidden_size, dropout=config.dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img, txt_tokens):
        img_tokens, _ = self.transformer(img)
        img_tokens = img_tokens.permute(0, 2, 1)
        img_tokens = self.img_token_proj(img_tokens)
        img_tokens = img_tokens.permute(0, 2, 1)

        txt_tokens = txt_tokens.permute(0, 2, 1)
        txt_tokens = self.txt_token_proj(txt_tokens)
        txt_tokens = txt_tokens.permute(0, 2, 1)

        cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        img_tokens = torch.cat([cls_token, img_tokens], dim=1)

        # txt_tokens = txt_tokens.permute(0, 2, 1)
        # txt_tokens = self.txt_token_proj(txt_tokens)
        # txt_tokens = txt_tokens.permute(0, 2, 1)

        img_tokens = self.img_pos_embedding(img_tokens)
        txt_tokens = self.txt_pos_embedding(txt_tokens)

        txt_tokens, _ = self.encoder(txt_tokens)
        img_tokens, attn_weights = self.decoder(img_tokens, txt_tokens)

        logits = self.head(img_tokens[:, 0])

        img_tokens = img_tokens[:, 1:].permute(0, 2, 1)
        img_tokens = self.txt_token_proj_inv(img_tokens)
        rec_txt_tokens = img_tokens.permute(0, 2, 1)

        return logits, attn_weights, rec_txt_tokens

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


class XFGConcatEncodedDR(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGConcatEncodedDR, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)
        self.transformer = Transformer(config)

        self.encoder = EncoderConcat(config)

        self.img_encoder = Encoder(config.encoder)
        self.txt_encoder = Encoder(config.encoder)

        self.img_token_proj = nn.Linear(config.visual_token_len, config.max_len - 1)

        self.head = nn.Linear(config.hidden_size, num_classes)

        self.pos_embedding = TrainablePositionalEncoding(config.max_len - 1 + config.max_len - 1, config.hidden_size, dropout=config.dropout)

        self.shared_cls = config.transformer.shared_cls

        if not self.shared_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img, txt_tokens):
        img_tokens, _ = self.transformer(img)

        img_tokens = img_tokens.permute(0, 2, 1)
        img_tokens = self.img_token_proj(img_tokens)
        img_tokens = img_tokens.permute(0, 2, 1)

        txt_tokens = txt_tokens[:, 1:, :]

        img_tokens, _ = self.img_encoder(img_tokens)
        txt_tokens, _ = self.txt_encoder(txt_tokens)

        tokens = torch.cat([txt_tokens, img_tokens], dim=1)
        tokens = self.pos_embedding(tokens) + tokens

        if not self.shared_cls:
            cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        else:
            cls_token = img_tokens[:, 0, :]

        tokens = torch.cat([cls_token, tokens], dim=1)

        tokens, attn_weights = self.encoder(tokens)

        logits = self.head(tokens[:, 0])
        return logits, attn_weights

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


class XFGConcatEncodedRec(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGConcatEncodedRec, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)
        self.transformer = Transformer(config)

        self.encoder = EncoderConcat(config)

        self.img_encoder = Encoder(config.encoder)
        self.txt_encoder = Encoder(config.encoder)

        self.img_token_proj = nn.Linear(config.visual_token_len, config.max_len - 1)

        self.head = nn.Linear(config.hidden_size, num_classes)

        self.pos_embedding = TrainablePositionalEncoding(config.max_len - 1 + config.max_len - 1, config.hidden_size, dropout=config.dropout)

        self.shared_cls = config.transformer.shared_cls

        if not self.shared_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img, txt_tokens):
        img_tokens, _ = self.transformer(img)

        img_tokens = img_tokens.permute(0, 2, 1)
        img_tokens = self.img_token_proj(img_tokens)
        img_tokens = img_tokens.permute(0, 2, 1)

        txt_tokens = txt_tokens[:, 1:, :]

        img_tokens, _ = self.img_encoder(img_tokens)
        txt_tokens, _ = self.txt_encoder(txt_tokens)

        tokens = torch.cat([txt_tokens, img_tokens], dim=1)
        tokens = self.pos_embedding(tokens) + tokens

        if not self.shared_cls:
            cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        else:
            cls_token = img_tokens[:, 0, :]

        tokens = torch.cat([cls_token, tokens], dim=1)

        tokens, attn_weights = self.encoder(tokens)

        logits = self.head(tokens[:, 0])
        return logits, attn_weights

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


class XFGConcatDR(nn.Module):
    def __init__(self, config, num_classes=200, zero_head=False):
        super(XFGConcatDR, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.dropout = nn.Dropout(config.dropout)
        self.transformer = Transformer(config)

        self.encoder = EncoderConcat(config)

        self.img_token_proj = nn.Linear(config.visual_token_len, config.max_len - 1)

        self.head = nn.Linear(config.hidden_size, num_classes)

        self.pos_embedding = TrainablePositionalEncoding(config.max_len - 1 + config.max_len - 1, config.hidden_size, dropout=config.dropout)

        self.shared_cls = config.transformer.shared_cls

        if not self.shared_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))


    def forward(self, img, txt_tokens):
        img_tokens, _ = self.transformer(img)

        img_tokens = img_tokens.permute(0, 2, 1)
        img_tokens = self.img_token_proj(img_tokens)
        img_tokens = img_tokens.permute(0, 2, 1)

        txt_tokens = txt_tokens[:, 1:, :]

        tokens = torch.cat([txt_tokens, img_tokens], dim=1)
        tokens = self.pos_embedding(tokens) + tokens

        if not self.shared_cls:
            cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        else:
            cls_token = img_tokens[:, 0, :]

        tokens = torch.cat([cls_token, tokens], dim=1)

        tokens, attn_weights = self.encoder(tokens)

        logits = self.head(tokens[:, 0])
        return logits, attn_weights

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
