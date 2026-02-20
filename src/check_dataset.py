from datasets import load_dataset

ds = load_dataset("delpgiero/customs-model1", split="train")
print(ds[0])
