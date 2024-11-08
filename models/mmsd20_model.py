from transformers import CLIPModel, CLIPVisionModel, AutoModel, AutoConfig, AutoTokenizer
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLayer
from models.visobert_model import TextModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = XLMRobertaLayer(config)  # Changed to XLMRobertaLayer for compatibility
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MV_CLIPVisoBert(nn.Module):
    def __init__(self, args):
        super(MV_CLIPVisoBert, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Load ViSoBERT with AutoModel and AutoConfig for compatibility
        self.visobert = TextModel("uitnlp/visobert")
        self.config = AutoConfig.from_pretrained("uitnlp/visobert")
        
        # Multimodal Encoder with XLMRobertaLayer
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)

        if args.simple_linear:
            self.text_linear = nn.Linear(args.text_size, args.text_size)
            self.image_linear = nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear = nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear = nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        self.classifier_fuse = nn.Linear(args.text_size, args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs_image, inputs_text, labels):
        # Obtain text and image features using ViSoBERT and CLIP
        visobert_outputs = self.visobert(input_ids=inputs_text['input_ids'], attention_mask=inputs_text['attention_mask'])
        text_features, text_feature = visobert_outputs

        clip_outputs = self.model(**inputs_image, output_attentions=True)
        image_features = clip_outputs['vision_model_output']['last_hidden_state']
        image_feature = clip_outputs['vision_model_output']['pooler_output']
        
        # Process text and image features
        image_feature = self.image_linear(image_feature)

        text_embeds = self.model.visual_projection(text_features)
        image_embeds = self.model.visual_projection(image_features) 
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        
        img_attention_mask = torch.ones((inputs_image['input_ids'].shape[0], image_features.shape[1])).to(image_features.device)
        attention_mask = torch.cat((img_attention_mask, inputs_text['attention_mask']), dim=-1)
        print(attention_mask.shape)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        print(extended_attention_mask.shape)
        
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs_text['input_ids'].device),
            inputs_text['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1), dim=-1)
        tw, iw = att.split([1, 1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature
        

        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)

        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        score = fuse_score + text_score + image_score
        outputs = (score,)

        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss = loss_fuse + loss_text + loss_image

            outputs = (loss,) + outputs
        return outputs
