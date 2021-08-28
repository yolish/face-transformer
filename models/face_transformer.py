"""
The Multi-Property FaceTransformer model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from util import box_ops

class FaceTransformer(nn.Module):
    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = False
        self.properties = config.get("properties")
        num_properties = len(self.properties)
        self.backbone = build_backbone(config)
        self.transformer = Transformer(config)
        decoder_dim = self.transformer.d_model
        self.input_proj = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_properties, decoder_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)


        self.binary_cls_head = nn.Linear(decoder_dim, 1)
        reg_heads = []
        multi_cls_heads = []
        self.prop_to_reg_head_idx = {}
        self.prop_to_cls_head_idx = {}
        for i, (property, property_dict) in enumerate(self.properties.items()):
            property_dim = property_dict["dim"]
            property_type = property_dict["type"]
            if property_type == "binary_cls":
                pass
            elif property_type == "multi_cls":
                j = len(self.prop_to_cls_head_idx)
                self.prop_to_cls_head_idx[property] = j
                multi_cls_heads.append(nn.Linear(decoder_dim, property_dim))
            elif property_type == "sigmoid_regression":
                j = len(self.prop_to_reg_head_idx)
                self.prop_to_reg_head_idx[property] = j
                reg_heads.append(PropertyRegressor(decoder_dim, property_dim))
            else:
                raise NotImplementedError("property type {} for property {} not supported".format(property_type, property))
        self.multi_cls_heads = nn.Sequential(*multi_cls_heads)
        self.reg_heads = nn.Sequential(*reg_heads)

    def forward_transformer(self, samples):
        """
        Forward of the Transformers
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return the latent embedding from the decoder
        """
        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract the features and the position embedding from the visual backbone
        features, pos = self.backbone(samples)

        src, mask = features[0].decompose()

        # Run through the transformer to translate to global and local properties
        assert mask is not None
        latent_property = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[0])[0][0]

        return latent_property

    def forward(self, data):
        """ The forward pass expects a dictionary with the following keys-values
         'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED


        returns a dictionary with the following keys-values: propeties name and value
        """
        latent_property = self.forward_transformer(data)
        res = {}
        for i, (property, property_dict) in enumerate(self.properties.items()):
            property_type = property_dict["type"]
            if property_type == "binary_cls":
                res[property] = self.binary_cls_head(latent_property[:, i, :])
            elif property_type == "multi_cls":
                j = self.prop_to_cls_head_idx.get(property)
                res[property] = self.multi_cls_heads[j](latent_property[:, i, :])
            elif property_type == "sigmoid_regression":
                j = self.prop_to_reg_head_idx.get(property)
                res[property] = self.reg_heads[j](latent_property[:, i, :])
        return res


class PropertyRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the output dimension

        """
        super().__init__()

        if output_dim == -1:
            self.just_normalize = True
        else:
            self.fc_h = nn.Linear(decoder_dim, decoder_dim*2)
            self.fc_o = nn.Linear(decoder_dim*2, output_dim)
            self.just_normalize = False
            self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.just_normalize:
            return F.normalize(x, dim=1, p=2)
        else:
            x = F.gelu(self.fc_h(x))
            return self.fc_o(x)


class FaceAttrCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.properties = config.get("properties")
        self.weight_dict = config.get("losses_weights")
        if self.weight_dict is None:
            self.weight_dict = {}

    def forward(self, res, target_dict):
        out = {}
        for i, (property, logit) in enumerate(res.items()):
            property_type = self.properties.get(property).get("type")
            if property_type == "binary_cls":  # binary classifier
                out[property] = F.binary_cross_entropy_with_logits(logit, target_dict[property])
            elif property_type == "multi_cls":
                out[property] = F.nll_loss(logit, target_dict[property])
            elif property_type == "sigmoid_regression":
                out[property] = F.smooth_l1_loss(F.sigmoid(logit), target_dict[property])
            else:
                # TODO add support in forward pass for representation learning with triplet loss
                raise NotImplementedError("Property type {} not supported".format(property_type))
        loss = 0.0
        for k, v in out.items():
            w = self.weight_dict.get(k)
            if w is None:
                w = 1.0
            loss += w*v
        loss_dict = {k:v.item() for k,v in out.items()} # for debugging puprposes
        return loss, loss_dict

def postprocess(res, config, img_size):
    postprocess_dict = config.get("properties")
    out = {}
    for property, x in res.items():
        f = postprocess_dict.get(property).get("type")
        if f == "binary_cls":  # binary classifier
            property_th = 0.8
            out[property] = (F.sigmoid(x) > property_th).to(dtype=torch.long)
        elif f == "multi_cls":
            out[property] = torch.argmax(F.log_softmax(x, dim=1))
        elif f == "sigmoid_regression":
            x = F.sigmoid(x)
            img_h, img_w = img_size
            scale_fct = torch.Tensor([img_w, img_h,
                                      img_w, img_h,
                                      img_w, img_h,
                                      img_w, img_h,
                                      img_w, img_h]).to(x.device)
            out[property] = x*scale_fct
        else:
            raise NotImplementedError("Attribute type {} not supported".format(f))
    return out