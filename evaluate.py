# -*- coding: utf-8 -*-
"""
离线检索消融实验脚本（不依赖 DeepSeek API，仅评测"检索 / 重排"环节）

评测对象（四路对比，逐级叠加组件）：
    1. BM25            纯关键词检索（中文单字切分）
    2. Vector          纯向量语义检索（FAISS + bge-small-zh）
    3. Hybrid          BM25 + 向量加权融合（0.6*vec + 0.4*bm25），不重排
    4. Hybrid+Rerank   混合检索后再经规则重排（精确匹配/关键词重叠/名次/长度）

评测指标：
    Recall@1 / Recall@4 / Recall@10 —— 正确段落是否出现在 Top-K 中
    MRR@4                          —— 正确段落首次出现位置的倒数（越靠前越高）

运行方式（在项目根目录下）：
    python evaluate.py
"""

import sys
from pathlib import Path

# 保证项目根目录在模块搜索路径中，便于直接 `python evaluate.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows 控制台默认 GBK，统一成 UTF-8，避免打印中文/表格报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.rag.store import documents, vectorstore
from app.rag.retriever import HybridRetriever, _tokenize
from app.rag.reranker import RuleBasedReranker

# ----------------------------------------------------------------------
# 评测集：(问题, 正确段落中必然出现的"独特短语")
# 独特短语用于在 documents 中定位"标准答案段落（gold）"，不依赖固定下标。
# 题目覆盖全部 26 段知识文档，包含语义改写题（向量友好）与专有名词/数字题（BM25 友好）。
# ----------------------------------------------------------------------
EVAL_SET = [
    # ===== 1. 公司介绍 =====
    ("公司是什么时候成立的？总部在哪里？", "成立于2015年"),
    ("公司在海外哪个城市有研发中心？", "新加坡设有研发中心"),
    ("公司现在大概有多少员工？", "员工超过1200人"),
    # ===== 2. 企业云平台 =====
    ("企业云平台已经服务了多少家客户？", "500家企业客户"),
    ("企业云平台通过了哪些安全认证？", "ISO27001"),
    ("企业云平台承诺的服务可用性是多少？", "99.99%"),
    # ===== 3. AI 中台 =====
    ("AI 中台支持接入哪些主流大模型？", "通义千问"),
    ("AI 中台除了NLP还集成了哪些人工智能能力？", "知识图谱"),
    # ===== 4. 休假制度 =====
    ("入职满一年可以休几天年假？", "5天年假"),
    ("女员工的产假一共有多少天？", "产假128天"),
    ("男员工的陪产假是多少天？", "陪产假15天"),
    ("病假每年累计不能超过多少天？", "不超过30天"),
    # ===== 5. 报销流程 =====
    ("费用发生后需要在多久内提交报销申请？", "15个工作日内"),
    ("报销金额超过 5000 元需要谁审批？", "总监审批"),
    ("报销审批通过后一般多久打款？", "5个工作日内"),
    # ===== 6. IT 支持 =====
    ("IT 部门的内部热线分机号是多少？", "分机8000"),
    ("紧急 IT 问题要求多长时间内响应？", "30分钟内响应"),
    # ===== 7. 数据湖 =====
    ("数据湖底层基于什么存储技术？", "HDFS"),
    ("数据湖可以支撑多大的数据规模？", "PB级"),
    # ===== 8. 差旅政策 =====
    ("员工出差需要提前多久提交申请？", "提前3个工作日"),
    ("一线城市出差的住宿标准是多少？", "600元/晚"),
    ("出差期间每天的餐补是多少？", "餐补100元"),
    # ===== 9. 低代码平台 =====
    ("低代码平台内置了多少组件模板？", "200+组件"),
    ("低代码平台平均让应用交付周期缩短了多少？", "缩短60%"),
    # ===== 10. 行为规范 =====
    ("公司规定的工作时间是怎样的？", "9:00-18:00"),
    ("员工离职需要提前多少天申请？", "提前30天"),
    ("迟到早退每次扣罚多少钱？", "扣罚50元"),
    # ===== 11. 安全合规 =====
    ("生产环境的操作日志需要保留多久？", "保留6个月"),
    ("公司多久做一次外部渗透测试？", "外部渗透测试"),
    # ===== 12. 绩效管理 =====
    ("公司绩效结果最高等级是什么？占比限制多少？", "S级占比不超过10%"),
    ("绩效连续不达标会进入什么计划？", "绩效改进计划"),
    # ===== 13. 培训与发展 =====
    ("公司每年给员工多少培训学时？", "40学时"),
    ("新员工的导师每个月有多少津贴？", "导师津贴为每月500元"),
    ("外部认证考试报销每年上限是多少？", "报销上限为3000元"),
    # ===== 14. 采购管理 =====
    ("大额采购需要几家供应商比价？", "三家供应商比价"),
    ("采购申请审批通过后生成什么单据？", "采购订单（PO）"),
    # ===== 15. 会议室预订 =====
    ("会议室需要提前多久预约？", "提前2小时预约"),
    ("预订会议室后多久不到场会被自动释放？", "15分钟内未到场"),
    # ===== 16. 信息安全事件响应 =====
    ("发现安全事件后必须在多长时间内上报？", "1小时内"),
    ("紧急安全事件要求多久内闭环？", "4小时内闭环"),
    # ===== 17. 客户服务 SLA =====
    ("最高等级客户的工单响应时间是多少？", "尊享级客户"),
    ("年度SLA达标率低于多少可以申请减免？", "SLA达标率低于95%"),
    # ===== 18. 产品研发流程 =====
    ("公司一个迭代周期是多长？", "两周为一个迭代周期"),
    ("代码合并前需要几个人评审？", "代码评审（Code Review）"),
    ("正式发布在星期几？", "发布窗口固定为每周二和周四"),
    # ===== 19. 财务管理制度 =====
    ("公司实行什么样的预算管理方式？", "全面预算管理"),
    ("财务凭证需要保存多少年？", "保存期限为10年"),
    # ===== 20. 员工福利 =====
    ("公司除了五险一金还买了什么保险？", "商业补充医疗保险"),
    ("员工年度体检标准是多少钱？", "每人800元"),
    ("员工遇到重大困难可以申请什么基金？", "员工互助基金"),
    # ===== 21. 招聘流程 =====
    ("社会招聘整体周期控制在多久以内？", "3周以内"),
    ("核心岗位入职前需要完成什么调查？", "第三方背景调查"),
    # ===== 22. 知识产权 =====
    ("员工有发明想法时首先要提交什么？", "发明披露书"),
    ("发明专利的发明人奖励是多少钱？", "发明专利奖励5000元"),
    ("公司代码可以传到个人GitHub吗？", "个人GitHub"),
    # ===== 23. 数据备份与容灾 =====
    ("公司数据备份分几级？", "三级备份策略"),
    ("核心系统的恢复点目标RPO是多少？", "恢复点目标（RPO）为4小时"),
    ("多久做一次容灾切换演练？", "容灾切换演练"),
    # ===== 24. 网络访问规范 =====
    ("远程办公VPN用什么认证方式？", "双因素认证"),
    ("生产网只能通过什么设备访问？", "堡垒机"),
    # ===== 25. 代码评审与发布 =====
    ("代码变更必须通过什么方式提交？", "合并请求（Merge Request）"),
    ("合并请求需要几个人批准？", "两名同组成员的批准"),
    ("发布前必须附什么方案？", "回滚方案"),
    # ===== 26. 开源软件合规 =====
    ("商业产品中禁止引入什么类型的开源许可证？", "传染性许可证"),
    ("所有开源组件须录入什么清单？", "软件物料清单（SBOM）"),
]


def _doc_index_by_content(content: str) -> int:
    """根据正文内容找到它在 documents 列表中的下标。"""
    for i, d in enumerate(documents):
        if d.page_content == content:
            return i
    return -1


def gold_index(keyphrase: str) -> int:
    """根据"独特短语"定位标准答案段落（gold）的下标，找不到返回 -1。"""
    for i, d in enumerate(documents):
        if keyphrase in d.page_content:
            return i
    return -1


# ----------------------------------------------------------------------
# 四路检索：统一返回"document 下标"的排名列表（长度 10）
# ----------------------------------------------------------------------
def bm25_rank(retriever: HybridRetriever, question: str, k: int = 10):
    """纯 BM25 关键词检索。"""
    scores = retriever._bm25.get_scores(_tokenize(question))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[:k]


def vector_rank(question: str, k: int = 10):
    """纯向量语义检索（FAISS）。"""
    results = vectorstore.similarity_search_with_score(question, k=k)
    ranking = []
    for doc, _dist in results:  # FAISS 已按距离从小到大（相似度从高到低）返回
        idx = _doc_index_by_content(doc.page_content)
        if idx >= 0:
            ranking.append(idx)
    return ranking


def hybrid_rank(retriever: HybridRetriever, question: str, k: int = 10):
    """混合检索（BM25 + 向量加权融合），不经过重排。"""
    results = retriever.retrieve(question)  # 已返回融合排序后的 Top-10
    ranking = []
    for doc, _score in results:
        idx = _doc_index_by_content(doc.page_content)
        if idx >= 0:
            ranking.append(idx)
    return ranking


def hybrid_rerank_rank(retriever: HybridRetriever, reranker: RuleBasedReranker,
                        question: str, k: int = 10):
    """混合检索 + 规则重排。"""
    candidates = retriever.retrieve(question)            # 混合检索粗召回 Top-10
    reranked = reranker.rerank(question, candidates, top_k=k)  # 规则重排
    ranking = []
    for doc, _score in reranked:
        idx = _doc_index_by_content(doc.page_content)
        if idx >= 0:
            ranking.append(idx)
    return ranking


# ----------------------------------------------------------------------
# 指标计算
# ----------------------------------------------------------------------
def is_hit(ranking, gold, k) -> int:
    return 1 if gold in ranking[:k] else 0


def reciprocal_rank(ranking, gold, k) -> float:
    for i, idx in enumerate(ranking[:k]):
        if idx == gold:
            return 1.0 / (i + 1)
    return 0.0


def _percent(value: float, total: int) -> str:
    return f"{value / total * 100:.1f}%"


def main() -> None:
    retriever = HybridRetriever()
    reranker = RuleBasedReranker()

    methods = {
        "BM25":          lambda q: bm25_rank(retriever, q),
        "Vector":        lambda q: vector_rank(q),
        "Hybrid":        lambda q: hybrid_rank(retriever, q),
        "Hybrid+Rerank": lambda q: hybrid_rerank_rank(retriever, reranker, q),
    }

    stats = {name: {"r1": 0, "r4": 0, "r10": 0, "mrr4": 0.0}
             for name in methods}
    misses = {name: [] for name in methods}

    valid = 0
    skipped = []
    for question, keyphrase in EVAL_SET:
        gold = gold_index(keyphrase)
        if gold < 0:
            skipped.append((question, keyphrase))
            continue
        valid += 1
        for name, rank_fn in methods.items():
            ranking = rank_fn(question)
            stats[name]["r1"] += is_hit(ranking, gold, 1)
            stats[name]["r4"] += is_hit(ranking, gold, 4)
            stats[name]["r10"] += is_hit(ranking, gold, 10)
            stats[name]["mrr4"] += reciprocal_rank(ranking, gold, 4)
            if gold not in ranking[:4]:
                misses[name].append(question)

    print("=" * 72)
    print(f"检索消融实验结果（有效问题 {valid} 个，评测集共 {len(EVAL_SET)} 个）")
    print("=" * 72)
    header = f"{'方法':<16}{'Recall@1':>10}{'Recall@4':>10}{'Recall@10':>11}{'MRR@4':>10}"
    print(header)
    print("-" * 72)
    for name in methods:
        s = stats[name]
        print(f"{name:<16}{_percent(s['r1'], valid):>10}"
              f"{_percent(s['r4'], valid):>10}{_percent(s['r10'], valid):>11}"
              f"{s['mrr4'] / valid:>10.3f}")
    print("=" * 72)

    print("\n各路在 Top-4 未召回正确段落的问题：")
    for name in methods:
        if misses[name]:
            print(f"  [{name}] {len(misses[name])} 题未命中：")
            for q in misses[name]:
                print(f"      - {q}")
        else:
            print(f"  [{name}] Top-4 全部命中")

    if skipped:
        print('\n以下题目未能定位到 gold 段落（请检查"独特短语"）：')
        for q, key in skipped:
            print(f"      - {q}  (短语: {key})")


if __name__ == "__main__":
    main()
