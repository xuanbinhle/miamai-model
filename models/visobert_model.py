from torch import nn
from transformers import AutoModel, AutoConfig

class TextModel(nn.Module):
    def __init__(self, args, model_name="uitnlp/visobert"):
        super(TextModel, self).__init__()

        self.visobert = AutoModel.from_pretrained(model_name)
        self.pooler_linear = nn.Linear(self.visobert.config.hidden_size, args.text_size) # Convert shape
    
    def forward(self, input_ids, attention_mask):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        return outputs.last_hidden_state, self.pooler_linear(outputs.pooler_output)
