from huggingface_hub import snapshot_download

# repo_type: model, dataset
# local_dir: where to save the downloaded files
snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type='model', local_dir='./meta-llama/Llama-2-7b-hf')