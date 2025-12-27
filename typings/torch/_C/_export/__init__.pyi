class CppExportedProgram: ...

def deserialize_exported_program(serialized_program: str) -> CppExportedProgram:
    """deserialize_exported_program(arg0: str) -> torch._C._export.CppExportedProgram"""

def serialize_exported_program(cpp_exported_program: CppExportedProgram) -> str:
    """serialize_exported_program(arg0: torch._C._export.CppExportedProgram) -> str"""
