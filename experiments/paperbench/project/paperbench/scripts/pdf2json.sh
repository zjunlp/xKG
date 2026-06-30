# #!/bin/bash

# # 1. 启动 GROBID 并获取 PID
# (cd /disk/disk_20T/luoyujie/s2orc-doc2json/grobid-0.7.3 && ./gradlew run &)
# GROBID_PID=$!

# # 2. 等待服务启动 (根据机器性能调整时间)
# echo "等待 GROBID 启动... (60s)"
# sleep 60

# 3. 定义路径并创建目录 (修正了你命令中的路径)
S2ORC_DIR="/disk/disk_20T/luoyujie/s2orc-doc2json"
PDF_SOURCE_DIR="/disk/disk_20T/luoyujie/preparedness/project/paperbench/data/papers"
OUTPUT_DIR="${S2ORC_DIR}/output_dir/paper_coder"
mkdir -p "$OUTPUT_DIR"

# 4. 查找所有 PDF 并处理
# 使用 find ... -exec 更简洁高效
find "$PDF_SOURCE_DIR" -type f -name "*.pdf" -exec echo "正在处理: {}" \; -exec python3 "${S2ORC_DIR}/doc2json/grobid2json/process_pdf.py" \
    -i {} \
    -t "${S2ORC_DIR}/temp_dir/" \
    -o "$OUTPUT_DIR" \;

# 5. 清理
echo "处理完毕，正在停止 GROBID 服务 (PID: $GROBID_PID)。"
kill $GROBID_PID
