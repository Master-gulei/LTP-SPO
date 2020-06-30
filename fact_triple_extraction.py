#!/usr/bin/env python
# coding=utf-8
"""
文本中事实三元组抽取
python *.py input.txt output.txt begin_line end_line
"""

# model path
MODELDIR = "ltp_data"

import sys
import os
from collections import defaultdict
from itertools import groupby

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer
from nltk.parse import *

# print("正在加载LTP模型... ...")

segmentor = Segmentor()
# print(os.path.join(MODELDIR, "cws.model").replace('\\', '/'))
segmentor.load(os.path.join(MODELDIR, "cws.model").replace('\\', '/'))

postagger = Postagger()
postagger.load(os.path.join(MODELDIR, "pos.model").replace('\\', '/'))

parser = Parser()
parser.load(os.path.join(MODELDIR, "parser.model").replace('\\', '/'))

recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model").replace('\\', '/'))

# labeller = SementicRoleLabeller()
# labeller.load(os.path.join(MODELDIR, "srl/"))

# print("加载模型完毕。")

in_file_name = "input.txt"
out_file_name = "output.txt"
ltp_pos_list = ["a", "b", "c", "d", "e", "g", "h", "i", "j", "k", "m", "n", "nd", "nh", "ni",
                "nl", "ns", "nt", "nz", "o", "p", "q", "r", "u", "v", "wp", "ws", "x", "z"]

ltp_ner_list = ["O", "S", "B", "I", "E"]

ltp_par_dict = {"SBV": "主谓关系", "VOB": "动宾关系", "IOB": "间宾关系", "FOB": "前置宾语", "DBL": "兼语", "ATT": "定中关系",
                "ADV": "状中结构", "CMP": "动补结构", "COO": "并列关系", "POB": "介宾关系", "LAD": "左附加关系", "RAD": "右附加关系",
                "IS": "独立结构", "HED": "核心关系"}

if len(sys.argv) > 1:
    in_file_name = sys.argv[1]

if len(sys.argv) > 2:
    out_file_name = sys.argv[2]

if len(sys.argv) > 3:
    begin_line = int(sys.argv[3])

if len(sys.argv) > 4:
    end_line = int(sys.argv[4])


class extraction_start():
    def run(self, in_file_name, out_file_name, input_sentence=None):
        in_file = open(in_file_name, 'r', encoding="utf-8")
        out_file = open(out_file_name, 'a')
        spo_list = []
        line_index = 1
        sentence_number = 0
        if input_sentence:
            text_line = [input_sentence]
        else:
            text_line = in_file.readlines()
        for sentence in text_line:
            try:
                sentence_s, sentence_frag_dict = sentence_standardizing(sentence)
                words, postags, netags, arcs = get_processor(sentence_s)
                child_dict_list = build_parse_child_dict(words, postags, arcs)
                # print(child_dict_list)
                # draw_tree(words, postags, arcs)
                spo_dict = {}
                spo_dict[sentence] = fact_triple_extract(words, postags, netags, arcs, child_dict_list,
                                                         sentence_frag_dict, out_file)
                if spo_dict:
                    spo_list.append(spo_dict)
                out_file.flush()
            except:
                pass
            sentence_number += 1
            # if sentence_number % 50 == 0:
            #     print("%d done" % (sentence_number))
            line_index += 1
        in_file.close()
        out_file.close()
        return spo_list


def sentence_standardizing(sentence):
    sentence_frag_dict = {}
    substitute_by_list = ["GGG", "FFF", "EEE", "DDD", "CCC", "BBB", "AAA"]
    punctuation_1 = ["《", "<", "【", "[", "{", "（", "(", "“", "\"", "‘", "'"]
    punctuation_2 = ["》", ">", "】", "]", "}", "）", ")", "”", "\"", "’", "'"]
    while 1:
        substitute_item = ""
        substitute_by_item = ""
        for i in range(len(sentence)):
            start_index, end_index = 0, 0
            for punc1, punc2 in zip(punctuation_1, punctuation_2):
                if punc1 == sentence[i]:
                    start_index = i
                    end_index = sentence.index(punc2)
                    break
            if end_index > start_index:
                substitute_item = sentence[start_index:end_index + 1]
                substitute_by_item = substitute_by_list.pop()
                sentence_frag_dict[substitute_by_item] = substitute_item
                break
        if substitute_item:
            sentence = sentence.replace(substitute_item, substitute_by_item)
        else:
            break
    return sentence, sentence_frag_dict


def draw_tree(words, postags, arcs):
    arclen = len(arcs)
    conll = ""
    for i in range(arclen):
        if arcs[i].head == 0:
            arcs[i].relation = "ROOT"
        conll += "\t" + words[i] + "(" + postags[i] + ")" + "\t" + postags[i] + "\t" + str(arcs[i].head) + "\t" + arcs[
            i].relation + "\n"
    conlltree = DependencyGraph(conll)  # 转换为依存句法图
    tree = conlltree.tree()  # 构建树结构
    tree.draw()  # 显示输出的树


def get_processor(sentence):
    words = segmentor.segment(sentence)
    # replace_word=[(word,i) for word,i in enumerate(words) if word in substitute_by_list]
    # for item in replace_word:
    #     words[item[1]-1]+=item[0]
    #     del words[item[1]]
    # print("\t".join(words))
    postags = postagger.postag(words)
    # print("\t".join(postags))
    netags = recognizer.recognize(words, postags)
    # print("\t".join(netags))
    arcs = parser.parse(words, postags)
    # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    return words, postags, netags, arcs


def fact_triple_extract(words, postags, netags, arcs, child_dict_list, sentence_frag_dict, out_file):
    """
    对于给定的句子进行spo三元组抽取
    """

    def _check_group_by_netags(group_list, words, netags, arcs):
        new_group_list = []
        for (item, start_index, end_index) in group_list:
            last_start_index, last_end_index = 0, 0
            for i in range(start_index - 1, end_index):
                if "B" in netags[i]:
                    ne = netags[i].split("-")[-1]
                    last_start_index = i + 1
                    last_end_index = i + list(netags[i:]).index("E-" + ne) + 1
                    if last_end_index <= end_index:
                        last_end_index = end_index
                    last_item = "".join(words[last_start_index - 1:last_end_index])
                    new_group_list.append((last_item, last_start_index, last_end_index))
                    break
                elif "I" in netags[i]:
                    ne = netags[i].split("-")[-1]
                    last_start_index = len(netags[:i]) - list(netags[:i])[::-1].index("B-" + ne)
                    last_end_index = i + list(netags[i:]).index("E-" + ne) + 1
                    if last_end_index <= end_index:
                        last_end_index = end_index
                    last_item = "".join(words[last_start_index - 1:last_end_index])
                    new_group_list.append((last_item, last_start_index, last_end_index))
                    break
                elif "E" in netags[i]:
                    ne = netags[i].split("-")[-1]
                    last_start_index = len(netags[:i]) - list(netags[:i])[::-1].index("B-" + ne)
                    last_end_index = end_index
                    last_item = "".join(words[last_start_index - 1:last_end_index])
                    new_group_list.append((last_item, last_start_index, last_end_index))
                    break
                elif "S" in netags[i]:
                    pass

            if last_start_index == 0:
                last_start_index = start_index
                last_end_index = end_index
                new_group_list.append((item, last_start_index, last_end_index))
        return new_group_list

    ### 解决ATT，定中关系嵌套问题，获得定中组合
    ATT_original_group_list, ATT_group_list = [], []
    done_index = 0
    for i, arc1 in enumerate(arcs):
        done_index_list = []
        if i >= done_index:
            if arc1.relation == "ATT":
                for j, arc2 in enumerate(arcs[i + 1:]):
                    k = j + i + 1
                    done_index_list.append(arcs[k - 1].head)
                    if arc2.relation != "ATT":
                        break
                done_index = max(done_index_list)
                original_group = "".join(words[i:done_index])
                ATT_original_group_list.append((original_group, i, done_index))
                # RAD和量词的处理, 过滤“的”之类
                st_index = i
                for j in range(i, done_index):
                    if arcs[j].relation == "RAD" or postags[j] == "wp":
                        st_index = j + 1
                m_index = st_index
                for j in range(st_index, done_index):
                    if "m" in postags[j] and "n" not in postags[j + 1]:
                        m_index = j + 2
                # 单纯的RAD或量词
                if m_index >= done_index:
                    m_index = done_index - 2
                group = "".join(words[m_index:done_index])
                ATT_group_list.append((group, m_index + 1, done_index))
    # 利用实体识别矫正定中组合
    ATT_group_list = _check_group_by_netags(ATT_group_list, words, netags, arcs)

    # print(ATT_group_list)
    # print(child_dict_list)

    # 分类ATT为名词组合和动名组合
    def _classify_ATT(ATT_group_list):
        # 分类定中组合（分类成名词和动宾）
        ATT_v_group_list = []
        ATT_n_group_list = ATT_group_list
        for i, (group, start_index, end_index) in enumerate(ATT_group_list):
            # v_last_index是索引+1的结果
            try:
                v_last_index = end_index - list(postags[start_index - 1:end_index])[::-1].index("v")
            except:
                continue
            if postags[end_index - 1] == "v":
                continue
            v_relate_start_index_list, v_relate_end_index_list = [], []
            if v_last_index > start_index:
                v_relate_start_index_list = [j for j in range(start_index - 1, v_last_index - 1) if
                                             arcs[j].head == v_last_index]
            if not v_relate_start_index_list:
                v_relate_start_index = v_last_index - 1
            else:
                v_relate_start_index = v_relate_start_index_list[0]
            # 从最后一个不是v的修饰开始，如果没有则v之后全是
            v_relate_end_index_list = [j for j in range(v_last_index, end_index) if arcs[j].head == v_last_index]
            if not v_relate_end_index_list:
                o_relate_start_index = v_last_index
            else:
                o_relate_start_index = v_relate_end_index_list[-1] + 1
            # s_list=[words[j] for j,arc in enumerate(arcs) if arc.head==v_last_index and arc.relation=="SBV"]
            # RAD和量词的处理, 过滤“的”之类
            st_index = o_relate_start_index
            for j in range(o_relate_start_index, done_index):
                if arcs[j].relation == "RAD":
                    st_index = j + 1
            m_index = st_index
            for j in range(st_index, done_index):
                if "m" in postags[j] and "n" not in postags[j + 1]:
                    m_index = j + 2
            # 单纯的RAD或量词
            if m_index >= done_index:
                m_index = done_index - 2
            ATT_v_group_list.append(
                (i, "".join(words[v_relate_start_index:v_last_index]), "".join(words[m_index:end_index])))
            ATT_n_group_list[i] = ("".join(words[o_relate_start_index:end_index]), o_relate_start_index + 1, end_index)
        # print(ATT_group_list)
        return ATT_n_group_list, ATT_v_group_list

    def _get_SBV_groups(s_index_list, e_index):
        SBV_groups_list = []
        for s_index in s_index_list:
            if postags[e_index] == "v":  # 矫正
                SBV_groups_list.append((words[s_index - 1], words[e_index], s_index, e_index + 1))
        return SBV_groups_list

    def _get_ADV_groups(s_index_list, e_index):
        # start_index_list.sort()
        # ad=reduce(lambda x, y: x+y, [words[i-1] for i in start_index_list])
        # 区分修饰是名词还是其他的状中结构,名词涉及相关
        ADV_groups_list = []
        for s_index in s_index_list:
            if "n" in postags[s_index - 1] and postags[s_index - 1] != "nd":
                ADV_groups_list.append((words[s_index - 1], words[e_index], s_index, e_index + 1))
            else:
                if postags[e_index] == "v" and postags[s_index - 1] == "p":  # 矫正
                    ADV_groups_list.append((None, words[e_index] + words[s_index - 1], s_index, e_index + 1))
                else:
                    ADV_groups_list.append((None, words[e_index], s_index, e_index + 1))
        return ADV_groups_list

    def _get_VOB_groups(s_index_list, e_index):  # 反
        VOB_groups_list = []
        for s_index in s_index_list:
            if "n" in postags[s_index - 1] and postags[s_index - 1] != "nd":
                VOB_groups_list.append((words[e_index], words[s_index - 1], e_index + 1, s_index))
        return VOB_groups_list

    def _get_IOB_groups(s_index_list, e_index):  # 反,涉及相关（作为关系的条件）
        IOB_groups_list = []
        for s_index in s_index_list:
            if "n" in postags[s_index - 1] and postags[s_index - 1] != "nd":
                IOB_groups_list.append((words[e_index], words[s_index - 1], e_index + 1, s_index))
        return IOB_groups_list

    def _get_FOB_groups(s_index_list, e_index):  # 反
        FOB_groups_list = []
        for s_index in s_index_list:
            if ("n" in postags[s_index - 1] or "ws" in postags[s_index - 1]) and postags[s_index - 1] != "nd":
                FOB_groups_list.append((words[e_index], words[s_index - 1], e_index + 1, s_index))
        return FOB_groups_list

    def _get_POB_groups(s_index_list, e_index):
        POB_groups_list = []
        for s_index in s_index_list:
            if "n" in postags[s_index - 1] and postags[s_index - 1] != "nd":
                # 考虑CMP和ADV 宾语的不同，是否是实体识别的三种实体（人名、地名、机构名），若是，则属于相关
                if "N" in netags[s_index - 1]:
                    POB_groups_list.append((words[e_index], words[s_index - 1], e_index + 1, s_index))
                else:
                    POB_groups_list.append((None, words[s_index - 1], e_index + 1, s_index))
            else:
                # 这种是在句法中当名词，但在词性中当其他词性的情况
                POB_groups_list.append((words[e_index], words[s_index - 1], e_index + 1, s_index))
        return POB_groups_list

    def _get_CMP_groups(s_index_list, e_index):
        CMP_groups_list = []
        s_index = s_index_list[-1]
        CMP_groups_list.append((None, words[e_index] + words[s_index - 1], s_index, e_index + 1))
        return CMP_groups_list

    def _get_COO_groups(s_index_list, e_index):
        COO_groups_list = []
        for s_index in s_index_list:
            if postags[s_index - 1] == postags[e_index]:
                COO_groups_list.append((words[e_index], words[s_index - 1], e_index + 1, s_index))
        return COO_groups_list

    get_groups_function_dict = {
        "SBV": _get_SBV_groups,
        "ADV": _get_ADV_groups,
        "VOB": _get_VOB_groups,
        "IOB": _get_IOB_groups,
        "FOB": _get_FOB_groups,
        "POB": _get_POB_groups,
        "CMP": _get_CMP_groups,
        "COO": _get_COO_groups
    }

    def _get_ATT_group(s, s_index):
        for item in ATT_group_list:
            # item[2]+1
            if s_index in [i for i in range(item[1], item[2] + 1)]:
                s = item[0]
                s_index = item[-1]
        return s, s_index

    def _get_complete_verb(v, v_index):
        if "CMP" in syntax_group_result_dict:
            for CMP_group in syntax_group_result_dict["CMP"]:
                if CMP_group[-1] == v_index:
                    v = CMP_group[1]
        return v

    syntax_group_result_dict = defaultdict(list)
    for start_index, item in enumerate(child_dict_list):
        if item:
            for syntax in item:
                try:
                    syntax_group_result = get_groups_function_dict[syntax](item[syntax], start_index)
                    if syntax_group_result:
                        syntax_group_result_dict[syntax].extend(syntax_group_result)
                except:
                    pass
    # print(syntax_group_result_dict)

    spo_list = []
    # 开始依据语义规则对句法进行合并
    if syntax_group_result_dict:
        SV_list = []
        VS_index_dict = defaultdict(list)
        for syntax in syntax_group_result_dict:
            if syntax == "SBV":
                for SBV_group in syntax_group_result_dict[syntax]:
                    s, v, s_index, v_index = SBV_group[:]
                    v = _get_complete_verb(v, v_index)
                    s, s_index = _get_ATT_group(s, s_index)
                    SV_list.append((s, v, s_index, v_index))
                    if str(v_index) in VS_index_dict:
                        VS_index_dict[str(v_index)].append((s, s_index))
                    else:
                        VS_index_dict[str(v_index)] = [(s, s_index)]
        # 对COO里的V进行S判断，以补充COO里无直接主语的V
        if "COO" in syntax_group_result_dict:
            for COO_group in syntax_group_result_dict["COO"]:
                fu_str_index = str(COO_group[-2])
                zi_str_index = str(COO_group[-1])
                if fu_str_index in VS_index_dict and zi_str_index not in VS_index_dict:
                    v = COO_group[1]
                    v_index = COO_group[-1]
                    v = _get_complete_verb(v, v_index)
                    for s, s_index in VS_index_dict[fu_str_index]:
                        s, s_index = _get_ATT_group(s, s_index)
                        SV_list.append((s, v, s_index, v_index))
                        if str(v_index) in VS_index_dict:
                            VS_index_dict[str(v_index)].append((s, s_index))
                        else:
                            VS_index_dict[str(v_index)] = [(s, s_index)]
                # if fu_str_index in SV_index_dict and zi_str_index not in SV_index_dict:
                #     s = COO_group[1]
                #     s_index = COO_group[-1]
                #     s, s_index = _get_ATT_group(s, s_index)
                #     for v,v_index in SV_index_dict[fu_str_index]:
                #         v = _get_complete_verb(v, v_index)
                #         SV_list.append((s, v, s_index, v_index))
                #         if str(s_index) in SV_index_dict:
                #             SV_index_dict[str(s_index)].append((v, v_index))
                #         else:
                #             SV_index_dict[str(s_index)] = [(v, v_index)]

        done_v_index_list = []
        # 有直接或间接主语的三元组
        for s, v, s_index, v_index in SV_list:
            for syntax in syntax_group_result_dict:
                if syntax == "VOB":
                    for v1, b1, v1_index, b1_index in syntax_group_result_dict[syntax]:
                        if v1_index == v_index:
                            b1, b1_index = _get_ATT_group(b1, b1_index)
                            spo_list.append((s, v, b1, s_index, v_index, b1_index))
                            done_v_index_list.append(v_index)
                if syntax == "FOB":
                    for _, b1, v1_index, b1_index in syntax_group_result_dict[syntax]:
                        if v1_index == v_index:
                            b1, b1_index = _get_ATT_group(b1, b1_index)
                            spo_list.append((s, v, b1, s_index, v_index, b1_index))
                            done_v_index_list.append(v_index)
                if syntax == "IOB":
                    for _, b1, v1_index, b1_index in syntax_group_result_dict[syntax]:
                        if v1_index == v_index:
                            b1, b1_index = _get_ATT_group(b1, b1_index)
                            spo_list.append((s, "相关", b1, s_index, None, b1_index))
                if syntax == "CMP":
                    for mid_group in syntax_group_result_dict[syntax]:
                        mid1_index = 0
                        mid_v = ""
                        if mid_group[-1] == v_index:
                            mid1_index = mid_group[-2]
                            mid_v = mid_group[1]
                            if mid_group[0]:
                                spo_list.append((s, "相关", mid_group[0], s_index, None, mid_group[2]))
                        if "POB" in syntax_group_result_dict and mid1_index:
                            for _, b1, mid2_index, b1_index in syntax_group_result_dict["POB"]:
                                if mid2_index == mid1_index:
                                    b1, b1_index = _get_ATT_group(b1, b1_index)
                                    spo_list.append((s, mid_v, b1, s_index, mid_group[-1], b1_index))
                                    done_v_index_list.append(v_index)
                if syntax == "ADV":
                    for mid_group in syntax_group_result_dict[syntax]:
                        mid1_index = 0
                        mid_v = ""
                        if mid_group[-1] == v_index:
                            mid1_index = mid_group[-2]
                            mid_v = mid_group[1]
                            if mid_group[0]:
                                spo_list.append((s, "相关", mid_group[0], s_index, None, mid_group[2]))
                        if "POB" in syntax_group_result_dict and mid1_index:
                            for mid2, b1, mid2_index, b1_index in syntax_group_result_dict["POB"]:
                                if mid2_index == mid1_index:
                                    b1, b1_index = _get_ATT_group(b1, b1_index)
                                    if mid2:
                                        spo_list.append((s, "相关", b1, s_index, None, b1_index))
                                    else:
                                        spo_list.append((s, mid_v, b1, s_index, mid_group[-1], b1_index))
                                        done_v_index_list.append(v_index)
            # 对ATT里的动宾结构做分析
            for i, (ATT_item, ATT_start, ATT_end) in enumerate(ATT_original_group_list):
                if v_index in [index for index in range(ATT_start, ATT_end + 1)] and "n" in postags[ATT_end - 1] and \
                        postags[ATT_end - 1] != "nd":
                    if v_index not in done_v_index_list:
                        spo_list.append((s, v, ATT_group_list[i][0], s_index, v_index, ATT_group_list[i][-1]))
                        done_v_index_list.append(v_index)

        # 从VOB或FOB中获得剩余的三元组
        if "VOB" in syntax_group_result_dict:
            for v, o, v_index, o_index in syntax_group_result_dict["VOB"]:
                # 代表由词性判断错误导致的动宾结构主语缺失
                if v_index not in done_v_index_list:
                    item_list = []
                    for s, v1, s_index, v1_index in SV_list:
                        item_list.append((v1_index, v1, s, s_index))
                    item_list.sort()
                    v1_index, v1, s, s_index = [item for item in item_list if item[0] < v_index][-1]
                    o, o_index = _get_ATT_group(o, o_index)
                    if v1_index in done_v_index_list:
                        v = _get_complete_verb(v, v_index)
                        spo_list.append((s, v, o, s_index, v_index, o_index))
                        done_v_index_list.append(v_index)
                    else:
                        spo_list.append((s, v1, o, s_index, v1_index, o_index))
                        done_v_index_list.append(v1_index)
        if "FOB" in syntax_group_result_dict:
            for v, o, v_index, o_index in syntax_group_result_dict["FOB"]:
                # 代表由被动语态导致的前宾结构主语缺失
                if v_index not in done_v_index_list:
                    for syntax in syntax_group_result_dict:
                        if syntax == "ADV" or syntax == "CMP":
                            for mid1, v1, mid1_index, v1_index in syntax_group_result_dict[syntax]:
                                if v_index == v1_index:
                                    if "POB" in syntax_group_result_dict:
                                        for mid2, o1, mid2_index, o1_index in syntax_group_result_dict["POB"]:
                                            if mid2_index == mid1_index:
                                                o, o_index = _get_ATT_group(o, o_index)
                                                o1, o1_index = _get_ATT_group(o1, o1_index)
                                                spo_list.append((o, v1, o1, o_index, v1_index, o1_index))
                                                done_v_index_list.append(v1_index)
    assert_list = ["是", "为"]
    spo_copy_list = spo_list[:]
    add_spo_list = []
    relate_SO_list = []
    for i, (s, p, o, s_index, p_index, o_index) in enumerate(spo_list):
        # “是”“作为”等断言
        for asse in assert_list:
            if asse in p:
                for j, item in enumerate(spo_list):
                    if o_index == item[-1] and s_index != item[3]:
                        spo_copy_list[j] = (item[0], item[1], s, item[3], item[4], s_index)
        # COO中主宾的并列关系
        if "COO" in syntax_group_result_dict:
            for mid1, mid2, mid1_index, mid2_index in syntax_group_result_dict["COO"]:
                if s_index == mid1_index:
                    add_spo_list.append((mid2, p, o, mid2_index, p_index, o_index))
                elif o_index == mid1_index:
                    add_spo_list.append((s, p, mid2, s_index, p_index, mid2_index))
        # 对所有主语加入相关关系
        for s1, _, o1, s1_index, _, o1_index in spo_list:
            if s_index != s1_index and (s, s1) not in relate_SO_list and (s1, s) not in relate_SO_list:
                add_spo_list.append((s, "相关", s1, s_index, None, s1_index))
                relate_SO_list.append((s, s1))

    spo_list = spo_copy_list + add_spo_list
    spo_set = set([item[:3] for item in spo_list])
    spo_copy_set = spo_set
    for s, p, o in spo_set:
        # 还原AAA等代指
        for frag in sentence_frag_dict:
            if frag in s:
                spo_copy_set.remove((s, p, o))
                s = s.replace(frag, sentence_frag_dict[frag])
                spo_copy_set.add((s, p, o))
            if frag in o:
                spo_copy_set.remove((s, p, o))
                o = o.replace(frag, sentence_frag_dict[frag])
                spo_copy_set.add((s, p, o))
    spo_set = spo_copy_set
    # print(spo_set)
    return spo_set


def build_parse_child_dict(words, postags, arcs):
    """
    为句子中的每个词语维护一个保存句法依存儿子节点的字典
    """
    child_dict_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:
                item = arcs[arc_index].relation
                if item in child_dict:
                    child_dict[item].append(arc_index + 1)
                else:
                    child_dict[item] = []
                    child_dict[item].append(arc_index + 1)
        # if child_dict.has_key('SBV'):
        #    print(words[index],child_dict['SBV'])
        child_dict_list.append(child_dict)
    return child_dict_list


def complete_e(words, postags, child_dict_list, word_index):
    """
    完善识别的部分实体
    """
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict:
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])

    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict:
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict:
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix


if __name__ == "__main__":
    spo_list = extraction_start().run(in_file_name, out_file_name)
