import huggingface_hub
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import os 
import sys



def download_hf_patterns(repo_id: str, folder_pattern: str, local_dir: str, repo_type: str = 'model', token: str = None):
    """
    Downloads files and folders from a Hugging Face repository based on a glob pattern.

    Args:
        repo_id (str): The ID of the repository.
        folder_pattern (str): A glob pattern to specify which folders/files to download 
                              (e.g., 'checkpoints/step-*/**').
        local_dir (str): The local directory where the files should be saved.
        repo_type (str, optional): The type of the repository. Defaults to 'model'.
        token (str, optional): Hugging Face token for private repositories. Defaults to None.
    """
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    # The pattern is now directly provided by the user
    allow_patterns = [folder_pattern]
    
    print("=" * 80)
    print(f"[*] 开始基于模式的下载任务:")
    print(f"    > 仓库ID: {repo_id}")
    print(f"    > 仓库类型: {repo_type}")
    print(f"    > 本地目标目录: {local_dir}")
    print(f"    > 下载模式: {allow_patterns}")
    print("=" * 80)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            token=token,
            local_dir_use_symlinks=False
        )
        print("\n[SUCCESS] 下载完成!")
        print(f"文件已保存到: {os.path.abspath(local_dir)}")
        
    except HfHubHTTPError as e:
        print(f"\n[ERROR] 下载失败! 发生HTTP错误: {e}")
        # ... (error handling as before) ...
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 发生未知错误: {e}")
        sys.exit(1)