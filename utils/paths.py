import os.path
import git

def codebase_folder():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.working_tree_dir

def models_storage():
    codebase = codebase_folder()
    return os.path.join(codebase,"models","JPL_TRACE_GASES_MODELS")
