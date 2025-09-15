import abc
import importlib
import typing as t

class Dependency(abc.ABC):
    @abc.abstractmethod
    def check(self):
        ...

    @abc.abstractmethod
    def install_instructions(self) -> str:
        ...


class ImportDependency(Dependency):
    def __init__(self, ref: str, install: str) -> None:
        self.ref = ref
        self.install = install

    def check(self):
        importlib.import_module(self.ref)

    def install_instructions(self) -> str:
        return self.install


def check_dependencies(dependencies: t.Sequence[str], hook: str):
    if isinstance(dependencies, str):
        dependencies = (dependencies,)

    for dependency in dependencies:
        if (dep := _DEPENDENCIES.get(dependency)) is None:
            raise RuntimeError(f"Unknown dependency '{dependency}'. This is likely a bug in the hook declaration.")

        try:
            dep.check()
        except Exception as e:
            raise RuntimeError(
                f"Missing dependency '{dependency}' requried by hook '{hook}'.\n"
                f"To install: {dep.install_instructions()}"
            ) from e


_DEPENDENCIES = {
    'rsciio': ImportDependency('rsciio', "'pip install rosettasciio' or 'conda install rosettasciio'"),
}