import torch
import torch.nn as nn

import IPython


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int) -> None:
        """Initialize the Self Attention Module.

        Args:
            embed_size (int): The embedding size of the input.
            heads (int): The amount of heads to split the input into.

        Returns:
            None
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * self.heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Self Attention Module.

        Args:
            values (torch.Tensor): The values to be used in the attention.
            keys (torch.Tensor): The keys to be used in the attention.
            query (torch.Tensor): The query to be used in the attention.
            mask (torch.Tensor): The mask to be used in the attention.

        Returns:
            out (torch.Tensor): The output of the forward pass.
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        attention_filter = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            attention_filter = attention_filter.masked_fill(mask == 0, float("-1e20"))

        attention_filter = torch.softmax(
            attention_filter / (self.embed_size ** (1 / 2)), dim=3
        )
        out = torch.einsum("nhql,nlhd->nqhd", [attention_filter, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_size: int, heads: int, dropout: float, forward_expansion: int
    ) -> None:
        """Initialize the Transformer Block.

        Args:
            embed_size (int): The embedding size of the input.
            heads (int): The amount of heads to split the input into.
            dropout (float): The dropout rate.
            forward_expansion (int): The expansion factor of the feed forward layer.

        Returns:
            None
        """
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask) -> torch.Tensor:
        """Forward pass of the Transformer Block.

        Args:
            value (torch.Tensor): The values to be used in the attention.
            key (torch.Tensor): The keys to be used in the attention.
            query (torch.Tensor): The query to be used in the attention.
            mask (torch.Tensor): The mask to be used in the attention.

        Returns:
            out (torch.Tensor): The output of the forward pass.
        """
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        device: torch.device,
        forward_expansion: int,
        dropout: float,
        max_length: int,
    ) -> None:
        """Initialize the Encoder.

        Args:
            src_vocab_size (int): The size of the source vocabulary.
            embed_size (int): The embedding size of the input.
            num_layers (int): The number of layers in the encoder.
            heads (int): The number of heads to split the input into.
            device (torch.device): The device to run the model on.
            forward_expansion (int): The expansion factor of the feed forward layer.
            dropout (float): The dropout rate.
            max_length (int): The maximum length of the input for position embedding.

        Returns:
            None
        """
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=embed_size
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=embed_size
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Encoder.

        Args:
            x (torch.Tensor): The input to the encoder.
            mask (torch.Tensor): The mask to be used in the attention.

        Returns:
            out (torch.Tensor): The output of the forward pass.
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        """Initialize the Decoder Block.

        Args:
            embed_size (int): The embedding size of the input.
            heads (int): The number of heads to split the input into.
            forward_expansion (int): The expansion factor of the feed forward layer.
            dropout (float): The dropout rate.
            device (torch.device): The device to run the model on.

        Returns:
            None
        """
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        key: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Decoder Block.

        Args:
            x (torch.Tensor): The input to the decoder block.
            value (torch.Tensor): The values to be used in the attention.
            key (torch.Tensor): The keys to be used in the attention.
            src_mask (torch.Tensor): The mask to be used in the attention.
            trg_mask (torch.Tensor): The mask to be used in the attention.

        Returns:
            out (torch.Tensor): The output of the forward pass.
        """
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        device: torch.device,
        forward_expansion: int,
        dropout: float,
        max_length: int,
    ) -> None:
        """Initialize the Decoder.

        Args:
            trg_vocab_size (int): The size of the target vocabulary.
            embed_size (int): The embedding size of the input.
            num_layers (int): The number of layers in the decoder.
            heads (int): The number of heads to split the input into.
            device (torch.device): The device to run the model on.
            forward_expansion (int): The expansion factor of the feed forward layer.
            dropout (float): The dropout rate.
            max_length (int): The maximum length of the input for position embedding.

        Returns:
            None
        """
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(
            num_embeddings=trg_vocab_size, embedding_dim=embed_size
        )
        self.position_embeding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=embed_size
        )
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Decoder.

        Args:
            x (torch.Tensor): The input to the decoder.
            enc_out (torch.Tensor): The output of the encoder.
            src_mask (torch.Tensor): The mask to be used in the attention.
            trg_mask (torch.Tensor): The mask to be used in the attention.

        Returns:
            out (torch.Tensor): The output of the forward pass.
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embeding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_pad_idx: int,
        trg_pad_idx: int,
        embed_size: int = 256,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 8,
        dropout: float = 0,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        max_length: int = 100,
    ) -> None:
        """Initialize the Transformer.

        Args:
            src_vocab_size (int): The size of the source vocabulary.
            trg_vocab_size (int): The size of the target vocabulary.
            src_pad_idx (int): The index of the padding token in the source vocabulary.
            trg_pad_idx (int): The index of the padding token in the target vocabulary.
            embed_size (int): The embedding size of the input.
            num_layers (int): The number of layers in the decoder.
            forward_expansion (int): The expansion factor of the feed forward layer.
            heads (int): The number of heads to split the input into.
            dropout (float): The dropout rate.
            device (torch.device): The device to run the model on.
            max_length (int): The maximum length of the input for position embedding.

        Returns:
            None
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Make the mask for the source.

        Args:
            src (torch.Tensor): The source input.

        Returns:
            src_mask (torch.Tensor): The mask for the source.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Make the mask for the target.

        Args:
            trg (torch.Tensor): The target input.

        Returns:
            trg_mask (torch.Tensor): The mask for the target.
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer.

        Args:
            src (torch.Tensor): The source input.
            trg (torch.Tensor): The target input.

        Returns:
            out (torch.Tensor): The output of the forward pass.
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
