"""Cluster mutations through the shared manager's locked write boundary."""

from sparkrun.api._context import resolve_sctx


def create_cluster(name: str, hosts: list[str], *, description: str = "", user: str | None = None, sctx=None):
    manager = resolve_sctx(sctx).cluster_manager
    manager.create(name, hosts, description=description, user=user)
    return manager.get(name)


def update_cluster(name: str, *, hosts: list[str] | None = None, description: str | None = None, user: str | None = None, sctx=None):
    manager = resolve_sctx(sctx).cluster_manager
    manager.update(name, hosts=hosts, description=description, user=user)
    return manager.get(name)


def delete_cluster(name: str, *, sctx=None):
    resolve_sctx(sctx).cluster_manager.delete(name)


def set_default_cluster(name: str | None = None, *, sctx=None):
    manager = resolve_sctx(sctx).cluster_manager
    manager.set_default(name) if name is not None else manager.unset_default()
