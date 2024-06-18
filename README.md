# RAG

------------------------------
最近LLM大模型异常火热，我判断RAG检索增强是未来的一个重要切入点，所以想试试。

主要的想法是利用我的微信公众号或利用gradio搭建网页，部署对话查询服务。LLM-EM将文献分段向量化存入向量数据库，当query来后，将query的embedding拿到向量数据库中检索出最相近的一条或几条，拼到prompt中喂给LLM-CHAT进行回答，并将检索结果以markdown引用形式拼在回答后面。

------------------------------

## 效果
### 不加ReRanker
```shell
<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant. 你是一个乐于助人的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。<|eot_id|>
<|start_header_id|>user<|end_header_id|>Question: 持有管制刀具怎么判？
Context: [Document(page_content='第二百九十七条 违反法律规定，携带武器、管制刀具或者爆炸物参加集会、游行、示威的，处三年以下有期徒刑、拘役、管制或者剥夺政治权利。'), Document(page_content='第一百三十条 非法携带枪支、弹药、管制刀具或者爆炸性、易燃性、放射性、毒害性、腐蚀性物品，进入公共场所或者公共交通工具，危及公共安全，情节严重的，处三年以下有期徒刑、拘役或者管制。'), Document(page_content='第一百二十八条 违反枪支管理规定，非法持有、私藏枪支、弹药的，处三年以下有期徒刑、拘役或者管制；情节严重的，处三年以上七年以下有期徒刑。依法配备公务用枪的人员，非法出租、出借枪支的，依照前款的规定处罚。依法配置枪支的人员，非法出租、出借枪支，造成严重后果的，依照第一款的规定处罚。单位犯第二款、第三款罪的，对单位判处罚金，并对其直接负责的主管人员和其他直接责任人员，依照第一款的规定处罚。')]<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>Answer:
根据中国《中华人民共和国刑法》第297条，违反相关法律规定而携带武器、管制刀具或爆炸物参与集会、游行、示威将被判处三年以下有期徒刑、拘役、管制或剥夺政治权利。

同时，《中华人民共和国刑法》的第一百三十条也明确了非法携带枪支、弹药、管制刀具等物资并且在公共场合或公共交通工具中进行，将面临三年以下有期徒刑、拘役或管制惩罚。

如果是私人拥有管制刀具的情况，则需要遵守相应的枪支管理规定，如果发现非法持有一定数量的管制刀具可能受到相应的刑事追究。具体情况可以参考《中华人民共和国刑法》的第一百二十八条关于非法持有人民兵装备以及私自出售军火的处理方法。
```

### 加ReRanker
```shell
<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant. 你是一个乐于助人的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。<|eot_id|>
<|start_header_id|>user<|end_header_id|>Question: 持有管制刀具怎么判？
Context: [Document(page_content='第二百九十七条 违反法律规定，携带武器、管制刀具或者爆炸物参加集会、游行、示威的，处三年以下有期徒刑、拘役、管制或者剥夺政治权利。'), Document(page_content='第三十八条 管制的期限，为三个月以上二年以下。判处管制，可以根据犯罪情况，同时禁止犯罪分子在执行期间从事特定活动，进入特定区域、场所，接触特定的人。对判处管制的犯罪分子，依法实行社区矫正。违反第二款规定的禁止令的，由公安机关依照《中华人民共和国治安管理处罚法》的规定处罚。'), Document(page_content='第一百三十条 非法携带枪支、弹药、管制刀具或者爆炸性、易燃性、放射性、毒害性、腐蚀性物品，进入公共场所或者公共交通工具，危及公共安全，情节严重的，处三年以下有期徒刑、拘役或者管制。')]<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>Answer:
持有管制刀具并且参与了非法行为（如聚众斗殴），将被视为犯罪，并可能面临三年以下有期徒刑或其他相应惩罚。此外，如果该人同时犯有其他罪名，则可以累加量刑。在某些特殊案例中，如涉嫌暴力恐怖主义等重大事件，被告方也可能因拥有管制刀具而受到更高程度的追究和谴责。
```

------------------------------

https://zhuanlan.zhihu.com/p/673552320

https://www.langchain.com.cn/getting_started/getting_started

https://llama-index.readthedocs.io/zh/latest/use_cases/queries.html

* 收集文献
    - [x] 法律条款
    - [ ] 菜谱
    - [ ] 小说（例如哈利波特等）
* 搭建服务
    - [x] 使用nvidia的docker
    - [x] llama有em模型吗？
    - [x] 处理文本，分段存储
    - [ ] 语义分割 <https://blog.csdn.net/v_JULY_v/article/details/135386202> （nlp_bert_document-segmentation_chinese-base）
    - [x] 向量化，存入向量数据库
    - [ ] 建图？用图检索？
    - [x] ranker
    - [ ] 增加bm25检索，用于多路召回。（加去重）
    - [x] 构造prompt模板，搭建推理服务
    - [ ] 部署在微信公众号或网页上
* 微调
    - [ ] 用LoRA微调专门的法律或小说插件
* 限时测试

------------------------------

未来
- [ ] 解析文档、pdf（llamaindex？pdfminer？RapidOCR？） <https://blog.csdn.net/v_JULY_v/article/details/135257848>
- [ ] 解析图片、表格
- [ ] 召回图片、表格
