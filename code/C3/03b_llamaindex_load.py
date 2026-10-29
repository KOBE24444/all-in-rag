
from pathlib import Path
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 使用与第一题相同的 Embedding 模型
# 查询向量必须与已保存的文档向量处于同一个语义空间
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 2. 定位第一题保存的索引目录
# __file__ 表示当前 Python 文件的位置
persist_dir = (
    Path(__file__).resolve().parent / "llamaindex_index_store"
)

# 3. 从本地 JSON 文件恢复存储结构
storage_context = StorageContext.from_defaults(
    persist_dir=str(persist_dir)
)

# 4. 恢复原有向量索引，不重新计算文档 Embedding
index = load_index_from_storage(storage_context)

print("向量索引加载成功！")

# 5. 创建检索器，指定返回最相关的两个节点
retriever = index.as_retriever(similarity_top_k=2)

# 6. 提出新的查询问题
question = "LlamaIndex有哪些功能？"

# 7. 查询会先转成向量，再与已有向量比较相似度
results = retriever.retrieve(question)

# 8. 打印检索结果和相似度得分
print(f"\n查询问题：{question}")

for i, result in enumerate(results, 1):
    print(f"\n检索结果 {i}：")
    print(result.node.get_content())
    print(f"相似度：{result.score}")
