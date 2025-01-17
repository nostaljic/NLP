import torch
import transformers
import datasets
from . import IValidService

class ArcValidService(IValidService):
    def get_next_topk_tokens(self, next_token_logits:torch.Tensor):
        next_token_probs = torch.softmax(next_token_logits, -1)
        next_token_topk = torch.topk(next_token_probs, 10)
        next_token_key_values = [
            (self.tokenizer.decode(idx), prob.item()) for idx, prob in zip(next_token_topk.indices, next_token_topk.values)
        ]
        return next_token_key_values
    
    def get_label_tokens(self, next_token_key_values:list, labels:list):
        label_probs = []
        next_token_key_values = dict(next_token_key_values)
        for l in labels:
            try:
                label_probs.append((l,next_token_key_values[l]))
            except:
                pass
        label_probs.sort(key= lambda x:-x[1])
        return label_probs

    def validate(self) -> None:
        for i,d in enumerate(self.dataset):
            input_ids = self.tokenizer.apply_chat_template(
                conversation=[
                    {"role":"user","content":d['system_prompt']},
                    {"role":"user","content":d['user_prompt']}], 
                add_generation_prompt=True, 
                tokenize=True,
                return_tensors='pt'
            )
            next_token_key_values = self.get_next_topk_tokens(
                self.model(input_ids).logits[0][-1], 
            )
            answer = self.get_label_tokens(next_token_key_values,d['label'])[0]

            answer_count = 0
            if answer[0] == d['real_answer']:
                answer_count+=1
                print(f"{i}. Output: {answer[0]} , Reference: {d['real_answer']} :: O")
            else:
                print(f"{i}. Output: {answer[0]} , Reference: {d['real_answer']} :: X")
        print(f"Score: {answer_count/len(self.dataset}")
                
    def preprocess_data(self, row) -> dict:
        common_system_prompt = "Select one answer from above candidates."
        #####
        row_question = row['question']
        row_labels = row['choices']['label']
        row_candidates = row['choices']['text']
        row_candidate_label_prompt = "\n".join([". ".join(c) for c in list(zip(row_labels, row_candidates))])
        row_user_prompt = "\n".join([row_question, row_candidate_label_prompt])
        #####
        row_answer = row['answerKey']
        return {
            "system_prompt":common_system_prompt,
            "user_prompt":row_user_prompt,
            "label":row_labels,
            "real_answer":row_answer
        }
        
