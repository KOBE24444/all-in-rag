
from pathlib import Path
from collections import Counter
from time import perf_counter
from unstructured.partition.pdf import partition_pdf

# 获取项目中原有 PDF 的绝对路径
pdf_path = Path(__file__).resolve().parents[2] / "data/C2/pdf/rag.pdf"

# 两种策略使用完全相同的 PDF
for strategy in ["hi_res", "ocr_only"]:
    print(f"\n========== {strategy} ==========")

    try:
        start = perf_counter()

        # 核心实验：仅改变解析策略
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=strategy,
            languages=["chi_sim", "eng"]
        )

        elapsed = perf_counter() - start

        # 比较解析耗时、元素数量与分类
        print("耗时（秒）：", round(elapsed, 2))
        print("元素数量：", len(elements))
        print("字符总数：", sum(len(str(e)) for e in elements))

        types = Counter(e.category for e in elements)
        print("元素类型分布：", dict(types))

        # 只显示前 6 个元素，避免终端输出过长
        for i, element in enumerate(elements[:6], 1):
            print(f"\n元素 {i}：{element.category}")
            print(str(element)[:200])

    except Exception as e:
        print("解析失败：", type(e).__name__)
        print(str(e)[:500])
