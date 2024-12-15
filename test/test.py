from utils_rag.EnhancedLocalEmbeddings import EnhancedLocalEmbeddings

local_embeddings = EnhancedLocalEmbeddings(model_path="../../acge_text_embedding")

embedding = local_embeddings.embed_text("The quick brown fox jumps over the lazy dog.")
print("单文本嵌入结果 | Single text embedding:", embedding[:10])  # 打印前 10 个维度