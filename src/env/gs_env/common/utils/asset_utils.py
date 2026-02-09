from pathlib import Path

from huggingface_hub import snapshot_download


def get_asset_path(
    asset_name: str,
    dataset_name: str = "Genesis-Intelligence/internal_assets",
) -> str:
    """
    Download the asset from Hugging Face Hub and return the local path.
    """
    asset_path = snapshot_download(
        repo_type="dataset", repo_id=dataset_name, allow_patterns=f"{asset_name}/*"
    )
    return f"{asset_path}/{asset_name}"


def get_mesh_path(name: str) -> str:
    asset_path = get_asset_path(
        asset_name=name,
        dataset_name="Genesis-Intelligence/assets",
    )
    return f"{asset_path}"


def get_urdf_path(name: str, end_effector_name: str) -> str:
    if name == "piper":
        asset_path = get_asset_path(asset_name="piper_description")
        if end_effector_name == "pika":
            urdf_path = f"{asset_path}/urdf/piper_with_pika_description.urdf"
        elif end_effector_name == "piper":
            urdf_path = f"{asset_path}/urdf/piper_description.urdf"
        else:
            raise ValueError(f"Unknown end effector name: {end_effector_name}")
        return urdf_path
    elif name == "ur5e":
        asset_path = get_asset_path(asset_name="ur_description")
        urdf_path = f"{asset_path}/urdf/ur5e.urdf"
        return urdf_path
    else:
        raise ValueError(f"Unknown robot name: {name}")


def resolve_file_path(file_path: str) -> str:
    """Resolve file path, trying the original path first, then relative to GenesisPlayground root."""
    path = Path(file_path)
    if not path.exists():
        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / path).exists():
                return str(current / file_path)
            current = current.parent
    return file_path
