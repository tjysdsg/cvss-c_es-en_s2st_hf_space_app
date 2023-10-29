def download_model(tag: str, out_dir: str):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=tag, local_dir=out_dir)
