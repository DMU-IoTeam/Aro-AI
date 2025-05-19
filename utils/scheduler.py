from transformers import get_cosine_schedule_with_warmup

def get_scheduler(optimizer, train_steps, warmup_steps):
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)


