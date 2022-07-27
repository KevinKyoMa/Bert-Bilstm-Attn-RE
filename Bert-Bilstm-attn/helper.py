import json
def prepare_data():
    #将开源的百度数据集处理为相应的json格式
    #数据使用的是百度发布的DUIE数据，包含了实体识别和关系抽取，原数据地址：https://ai.baidu.com/broad/download?dataset=dureader
    print("---Regenerate Data---")
    with open("./data/train_data.json", 'r', encoding='utf-8') as load_f:
        info = []
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data = {}
                single_data['rel'] = j["predicate"]
                single_data['ent1'] = j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text'] = dic['text']
                info.append(single_data)
        sub_train = info
    with open("./data/train.json", "w", encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")

    with open("./data/dev_data.json", 'r', encoding='utf-8') as load_f:
        info = []
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data = {}
                single_data['rel'] = j["predicate"]
                single_data['ent1'] = j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text'] = dic['text']
                info.append(single_data)

        sub_train = info
    with open("./data/dev.json", "w", encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")