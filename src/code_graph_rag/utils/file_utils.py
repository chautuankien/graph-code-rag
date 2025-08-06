import os 
from pathlib import Path

def is_package(folder_path: Path) -> bool:
    """A package is a folder contains __init__.py"""
    return (folder_path / "__init__.py").is_file()

def list_all_files(repo_path: Path) -> list[Path]:
    """Find all files in a repo"""
    return [p for p in repo_path.rglob("*") if p.is_file()]

def walk_codebase(repo_path: Path) -> dict:
    """
    Return a dict contains modules: folder, file, module, package
    """

    folders = []
    files = []
    packages = []
    modules = []

    for root, dirs, filenames in os.walk(repo_path):
        root_path = Path(root)

        # Folder
        if root_path != repo_path:
            folders.append({
                "id": f"folder:{root_path.relative_to(repo_path)}",
                "type": "folder",
                "path": str(root_path)
            })
        
        # Package
        if is_package(root_path):
            packages.append({
                "id": f"package:{root_path.relative_to(repo_path)}",
                "type": "Package",
                "path": str(root_path),
            })

        # Files
        for fname in filenames:
            file_path = root_path / fname
            rel_path = file_path.relative_to(repo_path)
            node = {
                "id": f"file:{rel_path}",
                "type": "File",
                "path": str(file_path),
            }
            files.append(node)

            # Modules
            if file_path.suffix == ".py":
                modules.append({
                    "id": f"module:{rel_path}",
                    "type": "Module",
                    "name": file_path.stem,
                    "path": str(file_path),
                })

    return {
        "folders": folders,
        "files": files,
        "modules": modules,
        "packages": packages,
    }
