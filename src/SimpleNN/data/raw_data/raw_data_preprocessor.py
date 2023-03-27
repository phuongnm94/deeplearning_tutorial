import json

from transformers import AutoTokenizer 

def get_speaker_name(s_id, gender):
    speaker = {
                "Ses01": {"F": "Mary", "M": "James"},
                "Ses02": {"F": "Patricia", "M": "John"},
                "Ses03": {"F": "Jennifer", "M": "Robert"},
                "Ses04": {"F": "Linda", "M": "Michael"},
                "Ses05": {"F": "Elizabeth", "M": "William"},
            }
    s_id_first_part = s_id[:5]
    return speaker[s_id_first_part][gender].upper()

def fix_around_window_collect(around_window = 1):

    for type_data in ["train", "valid", "test"]:
        flatten_data = []
        file_raw_data = f"iemocap.{type_data}.json"
        data = json.load(open(file_raw_data))

        print(len(data))
        for s_id, s_info in data.items():
            new_sentences = []
            for i, sentence in enumerate(s_info['sentences']):
                tmp_s = ""
                for j in range(max(0, i-around_window), min(len(s_info['sentences']), i+around_window+1)):
                    if i == j:
                        tmp_s += " </s>"
                    tmp_s +=  f" {get_speaker_name(s_id, s_info['genders'][j])}: {s_info['sentences'][j]}"
                    if i == j:
                        tmp_s += " </s>"
                new_sentences.append(tmp_s)

            flatten_data += list(zip(new_sentences, s_info['labels']))

            json.dump(flatten_data, open(file_raw_data.replace(".json", f"window{around_window}.flatten.json"), "wt"), indent=2)

def sentence_len_larger_limit(valid_s, bert_tokenizer):
    tokenized = bert_tokenizer(valid_s, truncation=False)
    return len(tokenized['input_ids']) > bert_tokenizer.model_max_length

def roberta_tokenize_limit_length_collect(pre_trained_model_name):
    bert_tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    max_around_window = 10
    for type_data in ["train", "valid", "test"]:
        flatten_data = []
        file_raw_data = f"iemocap.{type_data}.json"
        data = json.load(open(file_raw_data))

        print(len(data))
        for s_id, s_info in data.items():
            new_sentences = []
            for i, sentence in enumerate(s_info['sentences']):
                tmp_s = f"</s></s> {get_speaker_name(s_id, s_info['genders'][i])}: {s_info['sentences'][i]} </s></s>"
                valid_s = tmp_s
                for i_around in range(1, max_around_window):
                    if i-i_around>=0:
                        # add prev sentences in the context 
                        tmp_s =  f"{get_speaker_name(s_id, s_info['genders'][i-i_around])}: {s_info['sentences'][i-i_around]} " + tmp_s
                        if sentence_len_larger_limit(tmp_s, bert_tokenizer):
                            break
                        else:
                            valid_s = tmp_s

                    if i+i_around < len(s_info['sentences']):
                        # add next sentences in the context 
                        tmp_s = tmp_s + f" {get_speaker_name(s_id, s_info['genders'][i-i_around])}: {s_info['sentences'][i+i_around]}" 
                        if sentence_len_larger_limit(tmp_s, bert_tokenizer):
                            break
                        else:
                            valid_s = tmp_s
                    
                new_sentences.append(valid_s)

            flatten_data += list(zip(new_sentences, s_info['labels']))

            json.dump(flatten_data, open(file_raw_data.replace(".json", f".contextLimit512.json"), "wt"), indent=2)

if __name__=="__main__":
    # fix_around_window_collect(around_window=3)

    
    roberta_tokenize_limit_length_collect(pre_trained_model_name = 'roberta-base')

