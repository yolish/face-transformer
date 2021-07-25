"""
The Multi-Property FaceTransformer model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone

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
        self.transformer_t = Transformer(config)
        decoder_dim = self.transformer_t.d_model
        self.input_proj = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_properties, decoder_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)
        property_heads = {}
        for property, property_dict in self.properties.items():
            property_dim = property_dict["dim"]
            att_net = PropertyHead(decoder_dim, property_dim)
            property_heads[property] = att_net
        self.property_heads = property_heads

    def forward_transformer(self, data):
        """
        Forward of the Transformers
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return the latent embedding from the decoder
        """
        samples = data.get('img')

        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract the features and the position embedding from the visual backbone
        features, pos = self.backbone(samples)

        src, mask = features[0].decompose()

        # Run through the transformer to translate to global and local properties
        assert mask is not None
        latent_property = self.transformer_t(self.input_proj_t(src), mask, self.query_embed_t.weight, pos[0])[0][0]

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
        for i, property in enumerate(self.properties.keys()):
            res[property] = self.heads[i](latent_property[:, i, :])

        return res


class PropertyHead(nn.Module):
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
        property_losses = {}
        for property, property_dict in self.properties.items():
            property_type = property_dict["type"]
            if property_type == "binary_cls":  # binary classifier
                property_losses[property] = nn.BCEWithLogitsLoss()
            elif property_type == "multi_cls":
                property_losses[property] = nn.NLLLoss()
            elif property_type == "regresion":
                property_losses[property] = nn.MSELoss()
            elif property_type == "representation":  # recognition with Triplet Loss # TODO add support in forward pass
                property_losses[property] = nn.TripletMarginLoss(margin=0.25)
            else:
                raise NotImplementedError("Attribute type {} not supported".format(property_type))
        self.property_losses = property_losses
        self.weight_dict = config.get("losses_weights")

    def forward(self, res):
        out = {}
        for property, logit in res.items():
           out[property] = self.losses(logit)
        return out


def postprocess(res, config):
    for property, x in res.items():
        property_type = config.get(property)["property_type"]
        property_th = config.get(property)["threshold"]
        if property_type == "binary_cls":  # binary classifier
            return F.sigmoid(x) > property_th
        elif property_type == "multi_cls":
            return torch.argmax(F.softmax(x, dim=1))
        elif property_type == "regresion":
            return x
        else:
            raise NotImplementedError("Attribute type {} not supported".format(property_type))