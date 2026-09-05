"""Keep compiler caches on disk even when tests compile outside a trainer."""

from cleanrl.shared.runtime import configure_compile_cache


def pytest_configure(config):
    configure_compile_cache()
