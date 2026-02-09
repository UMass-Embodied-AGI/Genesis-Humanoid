from typing import Any, Protocol, TypeVar, runtime_checkable

from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint

T_cv = TypeVar("T_cv", covariant=True)


@runtime_checkable
class ToGSConvertible(Protocol[T_cv]):
    def to_gs(self) -> T_cv: ...


def to_gs_and_assert(obj: Any, expected_type: type) -> object:
    assert isinstance(obj, ToGSConvertible), (
        f"Object of type {type(obj).__name__} must have a 'to_gs()' method"
    )

    result = obj.to_gs()
    if not isinstance(result, expected_type):
        raise TypeError(
            f"to_gs() returned {type(result).__name__}, expected {expected_type.__name__}"
        )

    return result


def dofs_idx_to_name(entity: RigidEntity, dofs_idx_local: list[int]) -> list[str]:
    dofs_name = []
    for dof_idx_local in dofs_idx_local:
        dof_name = None
        for jnt_or_jnt_list in entity.joints:
            if isinstance(jnt_or_jnt_list, list):
                for jnt in jnt_or_jnt_list:
                    assert isinstance(jnt, RigidJoint)
                    if jnt.dof_idx_local == dof_idx_local:
                        dof_name = jnt.name
                        break
            else:
                assert isinstance(jnt_or_jnt_list, RigidJoint)
                if jnt_or_jnt_list.dof_idx_local == dof_idx_local:
                    dof_name = jnt_or_jnt_list.name
                    break
        assert dof_name is not None, f"Cannot find name for index {dof_idx_local}"
        dofs_name.append(dof_name)

    return dofs_name
