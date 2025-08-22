import torch
TENSOR_DICT = {
    "state_batch": torch.float32,
    "action_batch": torch.int64,
    "reward_batch": torch.float32,
    "new_state_batch": torch.float32,
    "done_batch": torch.float32,
}
