import torch
import re
from . import IValidService

class GSM8KValidService(IValidService):
    def get_next_topk_tokens(self, next_token_logits: torch.Tensor):
        next_token_probs = torch.softmax(next_token_logits, -1)
        next_token_topk = torch.topk(next_token_probs, 10)
        next_token_key_values = [
            (self.tokenizer.decode(idx), prob.item()) for idx, prob in zip(next_token_topk.indices, next_token_topk.values)
        ]
        return next_token_key_values
    
    def extract_final_answer(self, generated_text: str) -> str:
        """Extracts the numeric answer from the model's generated text."""
        match = re.search(r"\n####\s*(\d+)", generated_text)
        if match:
            return match.group(1)
        return None

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
            explanation = self.tokenizer.decode(
                self.model.generate(input_ids, max_length=512)[0], skip_special_tokens=True
            )

            input_with_final_prompt = explanation + "\n####"
            input_ids_with_final_prompt = self.tokenizer(
                input_with_final_prompt, return_tensors='pt'
            )
            final_answer_text = self.tokenizer.decode(
                self.model.generate(input_ids_with_final_prompt['input_ids'], max_length=512)[0],
                skip_special_tokens=True
            )

            predicted_answer = self.extract_final_answer(final_answer_text)

            if predicted_answer == d['real_answer']:
                answer_count += 1
                print(f"{i}. Output: {predicted_answer}, Reference: {d['real_answer']} :: O")
            else:
                print(f"{i}. Output: {predicted_answer}, Reference: {d['real_answer']} :: X")
        
        print(f"Score: {answer_count / len(self.dataset)}")
                
    def preprocess_data(self, row) -> dict:
        common_system_prompt = "Solve the problem and provide the numeric answer."
        #####
        row_question = row['question']
        row_real_answer = re.search(r"\n####\s*(\d+)", row['answer']).group(1)  # Extract numeric answer
        row_user_prompt = row_question  # The question itself is the user prompt
        
        return {
            "system_prompt": common_system_prompt,
            "user_prompt": row_user_prompt,
            "real_answer": row_real_answer
        }
