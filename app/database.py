import asyncio

active_sessions = {}

metadata_locks = {}

def get_metadata_lock(folder_path_str):
    if folder_path_str not in metadata_locks:
        metadata_locks[folder_path_str] = asyncio.Lock()
    return metadata_locks[folder_path_str]