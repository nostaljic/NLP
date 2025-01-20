import torch
from . import IValidService

class WinograndeValidService(IValidService):
    def get_next_topk_tokens(self, next_token_logits: torch.Tensor):
        next_token_probs = torch.softmax(next_token_logits, -1)
        next_token_topk = torch.topk(next_token_probs, 10)
        next_token_key_values = [
            (self.tokenizer.decode(idx), prob.item()) for idx, prob in zip(next_token_topk.indices, next_token_topk.values)
        ]
        return next_token_key_values
    
    def get_label_tokens(self, next_token_key_values: list, labels: list):
        label_probs = []
        next_token_key_values = dict(next_token_key_values)
        for l in labels:
            try:
                label_probs.append((l, next_token_key_values[l]))
            except:
                pass
        label_probs.sort(key=lambda x: -x[1])
        return label_probs

    def validate(self) -> None:
        answer_count = 0
        for i, d in enumerate(self.dataset):
            input_ids = self.tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": d['system_prompt']},
                    {"role": "user", "content": d['user_prompt']}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors='pt'
            )
            next_token_key_values = self.get_next_topk_tokens(
                self.model(input_ids).logits[0][-1],
            )
            answer = self.get_label_tokens(next_token_key_values, d['label'])[0]
            
            if answer[0] == d['real_answer']:
                answer_count += 1
                print(f"{i}. Output: {answer[0]} , Reference: {d['real_answer']} :: O")
            else:
                print(f"{i}. Output: {answer[0]} , Reference: {d['real_answer']} :: X")
        
        print(f"Score: {answer_count/len(self.dataset)}")
                
    def preprocess_data(self, row) -> dict:
        common_system_prompt = "Choose the most likely completion for the sentence."
        #####
        row_sentence = row['sentence']
        row_option1 = row['option1']
        row_option2 = row['option2']
        row_answer = row['answer']
        # Prepare the candidate prompt using the options
        row_choices_prompt = "\n".join([f"1. {row_option1}", f"2. {row_option2}"])
        row_user_prompt = f"{row_sentence} {row_choices_prompt}"
        
        ###### the correct label in numeric form (1 or 2)
        return {
            "system_prompt": common_system_prompt,
            "user_prompt": row_user_prompt,
            "label": ["1", "2"],  # The options A and B
            "real_answer": row_answer
        }
