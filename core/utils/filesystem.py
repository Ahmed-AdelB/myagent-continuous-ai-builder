"""
Filesystem utility functions for agents
"""

import os
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger


async def list_directory(path: str = ".", recursive: bool = False) -> List[Dict[str, Any]]:
    """
    List files and directories in a path.

    Args:
        path: Directory path to list (default: current directory)
        recursive: Whether to recurse into subdirectories

    Returns:
        List of file/directory information dicts with:
        - name: File/directory name
        - path: Full path
        - type: 'file' or 'directory'
        - size: File size in bytes (for files)
    """
    try:
        base_path = Path(path)

        if not base_path.exists():
            logger.error(f"Path does not exist: {path}")
            return []

        if not base_path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return []

        results = []

        if recursive:
            # Recursive listing using rglob
            for item in base_path.rglob('*'):
                if item.is_file():
                    results.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'file',
                        'size': item.stat().st_size
                    })
                elif item.is_dir():
                    results.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'directory',
                        'size': None
                    })
        else:
            # Non-recursive listing
            for item in base_path.iterdir():
                if item.is_file():
                    results.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'file',
                        'size': item.stat().st_size
                    })
                elif item.is_dir():
                    results.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'directory',
                        'size': None
                    })

        logger.debug(f"Listed {len(results)} items from {path}")
        return results

    except Exception as e:
        logger.error(f"Error listing directory {path}: {e}")
        return []


async def read_file(file_path: str) -> str:
    """
    Read contents of a file.

    Args:
        file_path: Path to file to read

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
    """
    try:
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        # Read file asynchronously
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, path.read_text)

        logger.debug(f"Read {len(content)} characters from {file_path}")
        return content

    except FileNotFoundError:
        raise
    except PermissionError as e:
        logger.error(f"Permission denied reading {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


async def write_file(file_path: str, content: str) -> bool:
    """
    Write content to a file, creating directories if needed.

    Args:
        file_path: Path to file to write
        content: Content to write

    Returns:
        True if successful

    Raises:
        PermissionError: If file can't be written
    """
    try:
        path = Path(file_path)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file asynchronously
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, path.write_text, content)

        logger.debug(f"Wrote {len(content)} characters to {file_path}")
        return True

    except PermissionError as e:
        logger.error(f"Permission denied writing to {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        raise


async def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: Path to check

    Returns:
        True if file exists
    """
    return Path(file_path).exists()


async def delete_file(file_path: str) -> bool:
    """
    Delete a file.

    Args:
        file_path: Path to file to delete

    Returns:
        True if successful
    """
    try:
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"Cannot delete non-existent file: {file_path}")
            return False

        if not path.is_file():
            logger.error(f"Cannot delete non-file: {file_path}")
            return False

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, path.unlink)

        logger.debug(f"Deleted file: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


async def create_directory(dir_path: str) -> bool:
    """
    Create a directory (and parent directories if needed).

    Args:
        dir_path: Path to directory to create

    Returns:
        True if successful
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created directory: {dir_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        return False
