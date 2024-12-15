# Utils

Mimic langchain to customize some tools for working with local models

## [Utils-RAG](./utils_rag/)

### [EnhancedLocalEmbeddings.py](./utils_rag/EnhancedLocalEmbeddings.py)

Here is an enhanced version of the LocalEmbeddings class that provides similar functionality to the OpenAI embedding model and includes more features and detailed comments in English and Chinese: | 以下是增强版的 LocalEmbeddings 类，它提供了与 OpenAI 嵌入模型类似的功能，并且包括更多特性和详细的中英文注释：

* **Initializing the Local Model | 初始化本地嵌入模型**

```python
# 1. 初始化本地嵌入模型 | Initialize local embedding model
# 使用 SentenceTransformer 模型
# Use a SentenceTransformer model
local_embeddings = EnhancedLocalEmbeddings(model_path="sentence-transformers/all-MiniLM-L6-v2")

# 使用 Hugging Face 模型（需要指定分词器路径）
# Use a Hugging Face model (requires tokenizer path)
local_embeddings_hf = EnhancedLocalEmbeddings(
    model_path="bert-base-uncased",
    tokenizer_path="bert-base-uncased"
)
```

* **Embedding Single Text | 嵌入单个文本**

```python
embedding = local_embeddings.embed_text("The quick brown fox jumps over the lazy dog.")
print("单文本嵌入结果 | Single text embedding:", embedding[:10])  # 打印前 10 个维度
```

* **Embedding Multiple Texts | 嵌入多个文本**

```python
texts = ["Text 1", "Text 2"]
embeddings = local_embeddings.embed_documents(texts)
print("多文本嵌入结果 | Multiple text embeddings:", [embedding[:10] for embedding in embeddings])
```

* **Embedding Single Texts Tsynchronously | 异步嵌入单个文本**

```python
import asyncio
async def async_example():
    embedding = await local_embeddings.aembed_text("Async embedding example.")
    print("异步单文本嵌入结果 | Async single text embedding:", embedding[:10])

asyncio.run(async_example())

```

* **Embedding Multipes Texts Tsynchronously | 异步嵌入单个文本**

```python
import asyncio

# 异步嵌入多个文本
async def async_example():
    texts = ["Text 1 for async.", "Text 2 for async."]
    embeddings = await local_embeddings.aembed_documents(texts)
    print("异步多文本嵌入结果 | Async multiple text embeddings:", [embedding[:10] for embedding in embeddings])

asyncio.run(async_example())

```

* **Embedded Query Text | 嵌入查询文本**

```python
query = "What is the meaning of life?"
query_embedding = local_embeddings.embed_query(query)
print("查询嵌入结果 | Query embedding:", query_embedding[:10])
```

* **Batch Embedding of Multiple Text | 批量嵌入多个文本**

```python
large_texts = ["Sample text " + str(i) for i in range(100)]
batch_embeddings = local_embeddings.embed_batch(large_texts, batch_size=16)
print("批量嵌入结果 | Batch embeddings:", [embedding[:10] for embedding in batch_embeddings[:3]])
```

* **Direct Call Example | 直接调用实例**

```python
texts = ["Direct call example 1.", "Direct call example 2."]
embeddings = local_embeddings(texts)
print("直接调用嵌入结果 | Direct call embeddings:", [embedding[:10] for embedding in embeddings])
```
