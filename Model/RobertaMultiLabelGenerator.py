import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput # 确保导入

class RobertaMultiLabelGenerator(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # 将7维输入嵌入投影到RoBERTa的隐藏维度
        self.input_projection = nn.Linear(7, config.hidden_size)

        # 输出头：直接输出27个值，每个值对应二进制向量的一个元素
        # 每个输出值是一个logit，代表该位置为1的“得分”
        self.classifier = nn.Linear(config.hidden_size, 27)

        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.Tensor = None, # 你的7维嵌入：(batch_size, 1024, 7)
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None, # 你的27维二进制向量标签：(batch_size, 27)
        return_dict: bool = True,
    ):
        # 1. 将7维嵌入投影到RoBERTa的隐藏维度
        projected_embeddings = self.input_projection(inputs_embeds)

        # 2. 将投影后的嵌入作为RoBERTa的输入
        outputs = self.roberta(
            inputs_embeds=projected_embeddings,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        # 3. 获取RoBERTa的输出（这里仍使用 [CLS] token 的表示）
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :] # 假设第一个token是 [CLS]

        # 4. 通过分类头进行预测，得到27个logits
        logits = self.classifier(pooled_output) # 形状为 (batch_size, 27)

        loss = None
        if labels is not None:
            # 使用 BCEWithLogitsLoss，它内部处理 Sigmoid
            loss_fct = nn.BCEWithLogitsLoss()
            # 确保 labels 是浮点型，因为 BCEWithLogitsLoss 期望目标是浮点型
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