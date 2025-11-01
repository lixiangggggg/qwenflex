import torch

# Replace with your actual model and router class imports
from model_wrappers import ElasticQwenForTraining
from router import ElasticRouter,BudgetEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
# Replace with the desired Qwen model name
model_name = "Qwen/Qwen-2.5-0.5B"
from config import (
    CKPT_PATH
)
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_full = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

# Move to GPU if available
device = "cuda2" if torch.cuda.is_available() else "cpu"
model_full = model_full.to(device)

question = "What is the capital of France?"
inputs = tokenizer(question, return_tensors="pt").to(device)
with torch.no_grad():
    output = model_full.generate(**inputs, max_new_tokens=128)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("Model_full's answer:", answer)

# Update to your actual checkpoint path
CHECKPOINT_PATH = CKPT_PATH

# Load the checkpoint (dictionary should contain keys for model and router)
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
# Initialize model and router
model_sub = ElasticQwenForTraining()
router = ElasticRouter()
encoder = BudgetEncoder()
# Load their state dicts from the checkpoint
model_sub.load_state_dict(checkpoint["model_state_dict"])
router.load_state_dict(checkpoint["router_state_dict"])
model_sub.eval()
router.eval()
# Example input, adjust shape/type as needed
buget = 0.5    
# Pass through router and then model
with torch.no_grad():
    enbeded_buget = encoder(buget)
    router_output = router(enbeded_buget)
    h_ratio = float(router_output["hidden_ratio"]
    ha_ratio = float(router_output["head_ratio"])
    inter_ratio = float(router_output["inter_ratio"])
    layer_mask = router_output["layer_mask"]
    current_tau = router_output["tau"]
    model_sub.set_active_subnet(
                h_ratio=h_ratio, 
                ha_ratio=ha_ratio, 
                intermediate_ratio=inter_ratio, 
                layer_mask=layer_mask
            )
    output_sub = model_sub.generate(**inputs, max_new_tokens=128)
  answer_sub = tokenizer.decode(output_sub[0], skip_special_tokens=True)
  print("Model_sub's answer:", answer_sub)
        


