"""Explicit cleanup of unused managed CSV/NPZ snapshots; never follows symlinks."""
from dataclasses import dataclass
import os
from pathlib import Path,PurePosixPath
import re
import stat

PREFIX='margam-owned:'
UUID=re.compile(r'[0-9a-f]{32}')
NAME=re.compile(r'[A-Za-z0-9_.-]{1,100}\.(csv|npz)')


def owned_references(value,workspace):
    """Collect mapping path values only; labels/expressions are never file refs."""
    result=set()
    def walk(item):
        if isinstance(item,dict):
            for key,child in item.items():
                if key=='data_fingerprints':continue # Only active mappings, never obsolete fingerprint bindings.
                if key=='path' and isinstance(child,str):
                    path=workspace.portable(child)
                    if path.startswith(PREFIX):
                        relative=path[len(PREFIX):]
                        parts=PurePosixPath(relative).parts
                        if not parts or relative.startswith('/') or '..' in parts or '\\' in relative:
                            raise ValueError('An owned mapping has an ambiguous path. Correct it before cleanup.')
                        result.add(PREFIX+PurePosixPath(relative).as_posix())
                else:walk(child)
        elif isinstance(item,(list,tuple)):
            for child in item:walk(child)
    walk(value)
    return result


def current_references(window):
    return owned_references([node.settings for node in window.scene.nodes()],window.project_workspace)


def history_references(stack):
    refs=set()
    def visit(command):
        nonlocal refs
        if command.childCount():
            return all(visit(command.child(i)) for i in range(command.childCount()))
        values=getattr(command,'resource_refs',None)
        if values is None:return False
        refs.update(values);return True
    return refs if all(visit(stack.command(i)) for i in range(stack.count())) else None


def protected_references(window):
    """Recompute before removal, including saved-on-disk discard state."""
    from .model_document import read_model
    workspace=window.project_workspace
    reasons={}
    def protect(values,reason):
        for value in values:reasons.setdefault(value,reason)
    protect(current_references(window),'Current model')
    if workspace.model_path is not None:
        protect(owned_references(read_model(workspace.model_path),workspace),'Saved model')
    for name in ('_trained_config','_evaluation_config'):
        config=getattr(window,name,None)
        if config is not None:
            protect(owned_references(dict(interior_data=config.interior_data,boundary_data=config.boundary_data),workspace),'Cached run')
    history=history_references(window.undo_stack)
    if history is None:raise ValueError('Some Undo commands have no file-reference record. Reopen the saved project before cleanup.')
    protect(history,'Undo / Redo')
    return reasons


@dataclass(frozen=True)
class Snapshot:
    relative: str
    size: int
    identity: tuple
    reason: str='Unused'

    @property
    def uri(self):return PREFIX+self.relative


def token(info):
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def existing_root(workspace):
    return workspace.companion(workspace.model_path) if workspace.model_path else Path(workspace._temporary.name) if workspace._temporary else None


def directory_fd(name,*,parent=None):
    if not all(hasattr(os,name) for name in ('O_NOFOLLOW','O_DIRECTORY')) or not all(operation in os.supports_dir_fd for operation in (os.open,os.stat,os.unlink,os.rmdir)) or os.listdir not in os.supports_fd:
        raise ValueError('Dataset cleanup requires descriptor-based filesystem operations on this platform.')
    return os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)


def scan_snapshots(workspace,reasons=None):
    """Recognise only UUID folders holding one regular, unlinked import file."""
    reasons={} if reasons is None else reasons
    root=existing_root(workspace)
    if root is None:return []
    rows=[]
    try:root_fd=directory_fd(root)
    except FileNotFoundError:return []
    try:
        try:data_fd=directory_fd('datasets',parent=root_fd)
        except FileNotFoundError:return []
        try:
            roots=(os.fstat(root_fd).st_dev,os.fstat(root_fd).st_ino,os.fstat(data_fd).st_ino)
            for folder in sorted(os.listdir(data_fd)):
                if not UUID.fullmatch(folder):continue
                try:folder_fd=directory_fd(folder,parent=data_fd)
                except OSError:continue
                try:
                    names=os.listdir(folder_fd)
                    if len(names)!=1 or not NAME.fullmatch(names[0]):continue
                    info=os.stat(names[0],dir_fd=folder_fd,follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:continue
                    relative='datasets/'+folder+'/'+names[0]
                    rows.append(Snapshot(relative,info.st_size,roots+(os.fstat(folder_fd).st_ino,)+token(info),reasons.get(PREFIX+relative,'Unused')))
                finally:os.close(folder_fd)
        finally:os.close(data_fd)
    finally:os.close(root_fd)
    return rows


def remove_snapshots(workspace,selected,reasons):
    """Recheck eligibility and inode metadata, then unlink through anchored FDs."""
    if workspace.busy:raise ValueError('Close the geometry editor before cleanup.')
    fresh={row.relative:row for row in scan_snapshots(workspace,reasons)}
    removed=[];kept=[];root=existing_root(workspace)
    if root is None:return [],[row.relative for row in selected]
    root_fd=directory_fd(root)
    try:
        data_fd=directory_fd('datasets',parent=root_fd)
        try:
            roots=(os.fstat(root_fd).st_dev,os.fstat(root_fd).st_ino,os.fstat(data_fd).st_ino)
            for row in selected:
                current=fresh.get(row.relative)
                if current is None or current.reason!='Unused' or current.identity!=row.identity:
                    kept.append(row.relative);continue
                _,folder,name=row.relative.split('/')
                try:
                    folder_fd=directory_fd(folder,parent=data_fd)
                    try:
                        info=os.stat(name,dir_fd=folder_fd,follow_symlinks=False)
                        identity=roots+(os.fstat(folder_fd).st_ino,)+token(info)
                        if identity!=row.identity or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or os.listdir(folder_fd)!=[name]:
                            kept.append(row.relative);continue
                        os.unlink(name,dir_fd=folder_fd);removed.append(row.relative)
                    finally:os.close(folder_fd)
                    try:os.rmdir(folder,dir_fd=data_fd)
                    except OSError:pass # Never remove extra files added concurrently.
                except OSError:kept.append(row.relative)
        finally:os.close(data_fd)
    finally:os.close(root_fd)
    return removed,kept
