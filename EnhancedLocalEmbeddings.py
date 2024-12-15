from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
import torch
from typing import Any, List, Optional
from sentence_transformers import SentenceTransformer

class EnhancedLocalEmbeddings(BaseModel, Embeddings):
    """
    本地嵌入模型类，支持 Hugging Face 和 SentenceTransformer。
    A class for local embedding models using Hugging Face or SentenceTransformer.
    """

    model_path: str = Field(..., description="本地模型路径 | Path to the local model directory.")
    tokenizer_path: Optional[str] = Field(None, description="本地分词器路径（可选）| Path to the local tokenizer directory (optional).")
    output_dim: Optional[int] = Field(None, description="输出嵌入维度（可选）| Dimension of the output embeddings (optional).")
    model: Optional[Any] = None  # 模型对象 | Model instance
    tokenizer: Optional[Any] = None  # 分词器对象 | Tokenizer instance

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型字段 | Allow arbitrary types

    def __init__(self, **data):
        """
        初始化模型。
        Initialize the embedding model.
        """
        super().__init__(**data)
        # 如果未提供分词器路径，假设使用 SentenceTransformer
        if not self.tokenizer_path:
            self.model = SentenceTransformer(self.model_path)
        else:
            # 加载 Hugging Face 模型和分词器
            self.model = AutoModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model.eval()  # 设置为评估模式 | Set the model to evaluation mode

    def embed_text(self, text: str) -> List[float]:
        """
        嵌入单个文本。
        Embed a single text using the local model.

        :param text: 要嵌入的文本 | The text to embed.
        :return: 文本的嵌入向量 | The embedding vector for the text.
        """
        if isinstance(self.model, SentenceTransformer):
            embeddings = self.model.encode(text, normalize_embeddings=True).tolist()
        else:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            tokens_tensor = inputs["input_ids"]

            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensor)
            # 取最后隐藏层的均值 | Take the mean of the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings[:self.output_dim] if self.output_dim else embeddings

    async def aembed_text(self, text: str) -> List[float]:
        """
        异步嵌入单个文本。
        Asynchronously embed a single text.

        :param text: 要嵌入的文本 | The text to embed.
        :return: 文本的嵌入向量 | The embedding vector for the text.
        """
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入多个文档。
        Embed a list of documents.

        :param texts: 文档列表 | A list of documents to embed.
        :return: 每个文档的嵌入向量列表 | A list of embedding vectors for the documents.
        """
        embeddings = [self.embed_text(text) for text in texts if text]
        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步嵌入多个文档。
        Asynchronously embed a list of documents.

        :param texts: 文档列表 | A list of documents to embed.
        :return: 每个文档的嵌入向量列表 | A list of embedding vectors for the documents.
        """
        return [await self.aembed_text(text) for text in texts if text]

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本。
        Embed a query text.

        :param text: 查询文本 | The query text to embed.
        :return: 查询文本的嵌入向量 | The embedding vector for the query.
        """
        return self.embed_text(text)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        使实例可调用，用于嵌入多个文本。
        Make the instance callable for embedding multiple texts.

        :param texts: 文本列表 | A list of texts to embed.
        :return: 文本嵌入向量的列表 | A list of embedding vectors for the texts.
        """
        return self.embed_documents(texts)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量嵌入多个文本，提高效率。
        Embed multiple texts in batches for better efficiency.

        :param texts: 文本列表 | A list of texts to embed.
        :param batch_size: 批量大小 | The batch size for embedding.
        :return: 文本嵌入向量的列表 | A list of embedding vectors for the texts.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if isinstance(self.model, SentenceTransformer):
                batch_embeddings = self.model.encode(batch, normalize_embeddings=True).tolist()
            else:
                inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = self.model(input_ids=inputs["input_ids"])
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
            embeddings.extend(batch_embeddings)
        return [embedding[:self.output_dim] if self.output_dim else embedding for embedding in embeddings]
