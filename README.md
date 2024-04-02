# RAG

------------------------------
最近LLM大模型异常火热，我判断RAG检索增强是未来的一个重要切入点，所以想试试。

主要的想法是利用我的微信公众号或利用gradio搭建网页，部署对话查询服务。LLM-EM将文献分段向量化存入向量数据库，当query来后，将query的embedding拿到向量数据库中检索出最相近的一条或几条，拼到prompt中喂给LLM-CHAT进行回答，并将检索结果以markdown引用形式拼在回答后面。

------------------------------

https://zhuanlan.zhihu.com/p/673552320

https://www.langchain.com.cn/getting_started/getting_started

* 收集文献
    - [ ] 法律条款
    - [ ] 菜谱
    - [ ] 小说（例如哈利波特等）
* 搭建服务
    - [ ] 使用nvidia的docker
    - [ ] llama有em模型吗？
    - [ ] 处理文本，分段存储
    - [ ] 向量化，存入向量数据库
    - [ ] 构造prompt模板，搭建推理服务
    - [ ] 部署在微信公众号或网页上
* 微调
    - [ ] 用LoRA微调专门的法律或小说插件
* 限时测试

------------------------------

未来
- [ ] 解析文档、pdf（llamaindex？pdfminer？）
- [ ] 解析图片、表格
- [ ] 召回图片、表格
