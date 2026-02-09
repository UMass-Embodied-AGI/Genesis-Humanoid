from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel


class EnvArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)


class OptitrackEnvArgs(EnvArgs):
    server_ip: str = "<server_ip>"
    client_ip: str = "<client_ip>"
    use_multicast: bool = False
    offset_config: str = "<optitrack_config>"
    tracked_link_names: list[str] = []
