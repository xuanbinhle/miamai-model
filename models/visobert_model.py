from torch import nn
from transformers import AutoModel, AutoConfig

class TextModel(nn.Module):
    def __init__(self, model_name="uitnlp/visobert"):
        super(TextModel, self).__init__()

        self.visobert = AutoModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output
