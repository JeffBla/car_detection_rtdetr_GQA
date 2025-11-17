from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torch.nn as nn


class RTDetrGQAForObjectDetection(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        id2label = {0: "car"}
        label2id = {"car": 0}

        self.model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r18vd",
            num_labels=1,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,  # 重新初始化分類 head
        )
        self.processor = RTDetrImageProcessor.from_pretrained(
            "PekingU/rtdetr_r18vd")

        # 3. 把 decoder 裡所有 self-attn 換成 GQA
        decoder = self.model.model.decoder  # 這個名稱可用 print(model) 確認

        if config["hidden_dim_GQA"] is not None:
            d_model = config["hidden_dim_GQA"]
        else:
            d_model = self.model.config.hidden_size  # 通常 256
        n_heads = self.model.config.num_attention_heads  # 通常 8

        for layer in decoder.layers:
            layer.self_attn = GroupedQueryAttention(
                embed_dim=d_model,
                num_q_heads=n_heads,
                num_kv_heads=config["num_kv_heads"],  # 推薦 4；你也可以試 2
                dropout=self.model.config.dropout,
            )

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )


# 2.  GQA 模組（介面像 HF attention：forward(hidden_states, attention_mask, ...)
class GroupedQueryAttention(nn.Module):

    def __init__(self, embed_dim, num_q_heads=8, num_kv_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.head_dim = embed_dim // num_q_heads

        self.q = nn.Linear(embed_dim, num_q_heads * self.head_dim)
        self.k = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.o = nn.Linear(num_q_heads * self.head_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x, B, N, n_heads):
        x = x.view(B, N, n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, N, Dh)

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                **kwargs):
        import math, torch
        B, N, C = hidden_states.shape

        q = self._reshape(self.q(hidden_states), B, N, self.num_q_heads)
        k = self._reshape(self.k(hidden_states), B, N, self.num_kv_heads)
        v = self._reshape(self.v(hidden_states), B, N, self.num_kv_heads)

        # KV head 分組 → repeat
        k = k.repeat_interleave(self.group_size, dim=1)  # (B, Hq, N, Dh)
        v = v.repeat_interleave(self.group_size, dim=1)  # (B, Hq, N, Dh)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.head_dim)
        if attention_mask is not None:
            # RT-DETR 的 mask 是 (B, 1, 1, N) 之類，直接加上去即可
            attn_scores = attn_scores + attention_mask

        attn = attn_scores.softmax(dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (B, Hq, N, Dh)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.o(context)

        if output_attentions:
            return out, attn
        return out, None
