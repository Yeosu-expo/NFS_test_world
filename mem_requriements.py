import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(model, 1, 5)
print("-------------------------------------------------------------------------")
deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(model, 1, 5)
deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(model, 5, 1)
