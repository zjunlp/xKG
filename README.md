# ResearchKG

- 创建`.env`文件并将`.env.example`中的内容拷贝过去并填充完
  
- cmd所在的目录：`ResearchKG`，一切命令都在这个目录下执行

- 执行的命令：`python -m xkg.run_kg` 模块化运行 以便适配到其它项目中

- 如何一键集成到外部：
  - 根据`xkg/interface/retrieve.py`中定义的接口 直接调用
  - 使用方法：`/disk/disk_20T/luoyujie/ResearchKG/xkg/run.py`中咋用就咋用 非常方便
  - 用的话直接import函数就行，类似：
    ```
    from ResearchKG.xkg.schema import Node
    from ResearchKG.xkg.interface.retrieve import (
        initialize_kg,
        find_node_by_paper_title,
        find_similar_techniques,
        find_similar_papers
    )
    ```
    