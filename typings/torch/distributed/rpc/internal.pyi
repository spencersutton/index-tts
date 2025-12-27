import pickle
from enum import Enum

__all__ = ["PythonUDF", "RPCExecMode", "RemoteException", "deserialize", "serialize"]
_thread_local_tensor_tables = ...
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler

class RPCExecMode(Enum):
    SYNC = ...
    ASYNC = ...
    ASYNC_JIT = ...
    REMOTE = ...

class _InternalRPCPickler:
    """
    This class provides serialize() and deserialize() interfaces to serialize
    data to be "binary string + tensor table" format
    So for RPC python UDF function and args, non tensor data will be serialized
    into regular binary string, tensor data will be put into thread local tensor
    tables, this serialization format is consistent with builtin operator and args
    using JIT pickler. This format will make tensor handling in C++ much easier,
    e.g. attach tensor to distributed autograd graph in C++
    """
    def __init__(self) -> None: ...
    def serialize(self, obj) -> tuple[bytes, list[Any]]:
        """
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
    def deserialize(self, binary_data, tensor_table) -> Any | AttributeError:
        """Deserialize binary string + tensor table to original obj"""

_internal_rpc_pickler = ...

def serialize(obj) -> tuple[bytes, list[Any]]: ...
def deserialize(binary_data, tensor_table) -> Any | AttributeError: ...

PythonUDF = ...
RemoteException = ...
