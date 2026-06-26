import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput # Ensure this is imported

class RobertaMultiLabelGenerator(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Project the 7-dimensional input embeddings to RoBERTa's hidden dimension
        self.input_projection = nn.Linear(7, config.hidden_size)

        # Output head: directly outputs 27 values, each corresponding to an element of the binary vector
        # Each output value is a logit, representing the "score" for that position being 1
        self.classifier = nn.Linear(config.hidden_size, 27)

        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.Tensor = None, # Your 7-dimensional embeddings: (batch_size, 1024, 7)
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None, # Your 27-dimensional binary vector labels: (batch_size, 27)
        return_dict: bool = True,
    ):
        # 1. Project the 7-dimensional embeddings to RoBERTa's hidden dimension
        projected_embeddings = self.input_projection(inputs_embeds)

        # 2. Pass the projected embeddings as input to RoBERTa
        outputs = self.roberta(
            inputs_embeds=projected_embeddings,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        # 3. Get RoBERTa's output (still using the [CLS] token representation here)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :] # Assuming the first token is [CLS]

        # 4. Pass through the classification head to get 27 logits
        logits = self.classifier(pooled_output) # Shape is (batch_size, 27)

        loss = None
        if labels is not None:
            # Use BCEWithLogitsLoss, which handles Sigmoid internally
            loss_fct = nn.BCEWithLogitsLoss()
            # Ensure labels are floats, as BCEWithLogitsLoss expects float targets
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
