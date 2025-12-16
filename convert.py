# pyright: reportUnknownVariableType=false, reportAny=false, reportUnknownArgumentType=false
from pathlib import Path
from typing import cast

import safetensors.torch
import torch
from torch import Tensor

base = Path("checkpoints")
SEPARATOR = "/__/"  # Use a separator unlikely to appear in key names


def flatten_tensors(
    d: dict[str, object] | list[object] | tuple[object, ...], prefix: str = ""
) -> tuple[dict[str, object], dict[str, str]]:
    """Flatten nested dict/list/tuple structure into flat dict with separator-separated keys.

    Returns:
        (tensors_dict, metadata_dict) where metadata contains type info for tuples and scalar values
    """
    result: dict[str, object] = {}
    metadata: dict[str, str] = {}

    if isinstance(d, (list, tuple)):
        is_tuple = isinstance(d, tuple)
        for idx, value in enumerate(d):
            full_key = f"{prefix}{idx}" if prefix else str(idx)
            if isinstance(value, (dict, list, tuple)):
                sub_tensors, sub_metadata = flatten_tensors(value, f"{full_key}{SEPARATOR}")
                result.update(sub_tensors)
                metadata.update(sub_metadata)
            elif isinstance(value, Tensor):
                result[full_key] = value
            elif value is None:
                metadata[full_key] = "none"
            elif isinstance(value, int):
                metadata[full_key] = f"int:{value}"
            elif isinstance(value, float):
                metadata[full_key] = f"float:{value}"
            elif isinstance(value, str):
                metadata[full_key] = f"str:{value}"
            else:
                print(f"WARNING: Skipping non-tensor key `{full_key}` of type {type(value)}")

        # Mark this collection's type
        collection_key = prefix.removesuffix(SEPARATOR) if prefix else "__root__"
        if is_tuple:
            metadata[f"__type__{collection_key}"] = "tuple"
        else:
            metadata[f"__type__{collection_key}"] = "list"
    else:
        for key, value in d.items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, (dict, list, tuple)):
                sub_tensors, sub_metadata = flatten_tensors(value, f"{full_key}{SEPARATOR}")
                result.update(sub_tensors)
                metadata.update(sub_metadata)
            elif isinstance(value, Tensor):
                result[full_key] = value
            elif value is None:
                metadata[full_key] = "none"
            elif isinstance(value, bool):
                metadata[full_key] = f"bool:{value}"
            elif isinstance(value, int):
                metadata[full_key] = f"int:{value}"
            elif isinstance(value, float):
                metadata[full_key] = f"float:{value}"
            elif isinstance(value, str):
                metadata[full_key] = f"str:{value}"
            else:
                print(f"WARNING: Skipping non-tensor key `{full_key}` of type {type(value)}")
    return result, metadata


def unflatten_tensors(flat_dict: dict[str, object], metadata: dict[str, str]) -> dict[str, object]:
    """Reconstruct original hierarchical structure from flattened dict using metadata."""
    result: dict[str, object] = {}

    # First pass: reconstruct structure with tensors and ints
    for key, value in flat_dict.items():
        parts = key.split(SEPARATOR)
        current: dict[str, object] | list[object] = result

        # Navigate/create nested structure
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, list):
                if not part.isdigit():
                    print(f"ERROR: Trying to navigate list with non-numeric key '{part}' in path '{key}'")
                    break
                idx = int(part)
                while len(current) <= idx:
                    # Check metadata to see if the next level should be a list
                    next_path_prefix = SEPARATOR.join(parts[: i + 1])  # Path up to current part
                    next_type_key = f"__type__{next_path_prefix}"
                    if next_type_key in metadata and metadata[next_type_key] in ("list", "tuple"):
                        current.append([])
                    else:
                        current.append({})
                next_current = current[idx]
                # If we encounter a non-container (like a Tensor), we can't navigate further
                if not isinstance(next_current, (dict, list)):
                    print(f"WARNING: Cannot navigate through '{key}' - encountered {type(next_current)} at index {idx}")
                    break
                current = cast(dict[str, object] | list[object], next_current)
            else:
                if part not in current:
                    # Check metadata to see if the next level should be a list
                    next_path_prefix = SEPARATOR.join(parts[: i + 1])  # Path up to current part
                    next_type_key = f"__type__{next_path_prefix}"
                    if next_type_key in metadata and metadata[next_type_key] in ("list", "tuple"):
                        current[part] = []
                    else:
                        current[part] = {}
                next_current = current[part]
                # If we encounter a non-container (like a Tensor), we can't navigate further
                if not isinstance(next_current, (dict, list)):
                    print(
                        f"WARNING: Cannot navigate through '{key}' - encountered {type(next_current)} at key '{part}'"
                    )
                    break
                current = cast(dict[str, object] | list[object], next_current)

        # Set the final value
        final_part = parts[-1]
        if isinstance(current, list):
            idx = int(final_part)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        elif isinstance(current, dict):
            current[final_part] = value
        else:
            # Current is probably a Tensor or other non-container type
            # This shouldn't happen in a well-formed flattening
            print(f"WARNING: Cannot set key '{key}' - parent is type {type(current)}, not a container")

    # Second pass: add int/float/str/None values from metadata
    for key, value in metadata.items():
        if key.startswith("__type__"):
            continue

        scalar_value: int | float | str | bool | None = None
        has_value = False
        if value == "none":
            scalar_value = None
            has_value = True
        elif value.startswith("bool:"):
            scalar_value = value[5:] == "True"
            has_value = True
        elif value.startswith("int:"):
            scalar_value = int(value[4:])
            has_value = True
        elif value.startswith("float:"):
            scalar_value = float(value[6:])
            has_value = True
        elif value.startswith("str:"):
            scalar_value = value[4:]
            has_value = True

        if not has_value:
            continue

        parts = key.split(SEPARATOR)
        current: dict[str, object] | list[object] = result

        for i, part in enumerate(parts[:-1]):
            if isinstance(current, list):
                if not part.isdigit():
                    print(f"ERROR: Trying to navigate list with non-numeric key '{part}' in metadata path '{key}'")
                    break
                idx = int(part)
                while len(current) <= idx:
                    # Check metadata to see if the next level should be a list
                    next_path_prefix = SEPARATOR.join(parts[: i + 1])
                    next_type_key = f"__type__{next_path_prefix}"
                    if next_type_key in metadata and metadata[next_type_key] in ("list", "tuple"):
                        current.append([])
                    else:
                        current.append({})
                current = cast(dict[str, object] | list[object], current[idx])
            else:
                if part not in current:
                    # Check metadata to see if the next level should be a list
                    next_path_prefix = SEPARATOR.join(parts[: i + 1])
                    next_type_key = f"__type__{next_path_prefix}"
                    if next_type_key in metadata and metadata[next_type_key] in ("list", "tuple"):
                        current[part] = []
                    else:
                        current[part] = {}
                current = cast(dict[str, object] | list[object], current[part])

        final_part = parts[-1]
        if isinstance(current, list):
            idx = int(final_part)
            while len(current) <= idx:
                current.append(None)
            current[idx] = scalar_value
        else:
            current[final_part] = scalar_value

    # Third pass: convert lists to tuples where indicated by metadata, and convert numeric string keys to integers
    def convert_tuples(
        obj: dict[str, object] | list[object], path: str = ""
    ) -> dict[str, object] | list[object] | tuple[object, ...]:
        if isinstance(obj, dict):
            # Convert numeric string keys to integers
            new_dict: dict[str | int, object] = {}
            for key, value in obj.items():
                # Try to convert key to int if it's a numeric string
                try:
                    int_key = int(key)
                    new_key: str | int = int_key
                except (ValueError, TypeError):
                    new_key = key

                current_path = f"{path}{SEPARATOR}{key}" if path else key
                if isinstance(value, (dict, list)):
                    new_dict[new_key] = convert_tuples(value, current_path)
                else:
                    new_dict[new_key] = value
            return new_dict
        if isinstance(obj, list):
            # First, recursively convert children
            converted_items = []
            for i, item in enumerate(obj):
                item_path = f"{path}{SEPARATOR}{i}" if path else str(i)
                if isinstance(item, (dict, list)):
                    converted_items.append(convert_tuples(item, item_path))
                else:
                    converted_items.append(item)

            # Then check if this list should be a tuple
            type_key = f"__type__{path}" if path else "__type____root__"
            if type_key in metadata and metadata[type_key] == "tuple":
                return tuple(converted_items)
            return converted_items
        return obj

    return cast(dict[str, object], convert_tuples(result))


def compare_structures(
    original: dict[str, object] | list[object] | tuple[object, ...],
    reconstructed: dict[str, object] | list[object] | tuple[object, ...],
    path: str = "",
) -> bool:
    """Compare two nested dict/list/tuple structures to ensure they match."""
    if isinstance(original, (list, tuple)) and isinstance(reconstructed, (list, tuple)):
        # Check type matches (list vs tuple)
        if type(original) is not type(reconstructed):
            print(f"Type mismatch at {path or 'root'}: {type(original).__name__} vs {type(reconstructed).__name__}")
            return False

        if len(original) != len(reconstructed):
            print(f"Length mismatch at {path or 'root'}: {len(original)} vs {len(reconstructed)}")
            return False

        for idx, (orig_val, recon_val) in enumerate(zip(original, reconstructed)):
            current_path = f"{path}[{idx}]" if path else f"[{idx}]"
            if isinstance(orig_val, (dict, list, tuple)) and isinstance(recon_val, (dict, list, tuple)):
                if not compare_structures(orig_val, recon_val, current_path):
                    return False
            elif isinstance(orig_val, Tensor) and isinstance(recon_val, Tensor):
                # Move both to CPU for comparison to handle device mismatches
                if not torch.equal(orig_val.cpu(), recon_val.cpu()):
                    print(f"Tensor mismatch at {current_path}: shapes={orig_val.shape} vs {recon_val.shape}")
                    return False
            elif orig_val is None and recon_val is None:
                pass  # Both are None, continue
            elif isinstance(orig_val, (int, float, str, bool)) and isinstance(recon_val, (int, float, str, bool)):
                if orig_val != recon_val:
                    print(f"Value mismatch at {current_path}: {orig_val} vs {recon_val}")
                    return False
            elif type(orig_val) is not type(recon_val):
                print(f"Type mismatch at {current_path}: {type(orig_val)} vs {type(recon_val)}")
                return False
        return True

    if isinstance(original, dict) and isinstance(reconstructed, dict):
        if set(original.keys()) != set(reconstructed.keys()):
            print(
                f"Key mismatch at {path or 'root'}: original={set(original.keys())}, "
                "reconstructed={set(reconstructed.keys())}"
            )
            return False

        for key, orig_val in original.items():
            current_path = f"{path}.{key}" if path else key
            recon_val = reconstructed[key]

            if isinstance(orig_val, (dict, list, tuple)) and isinstance(recon_val, (dict, list, tuple)):
                if not compare_structures(orig_val, recon_val, current_path):
                    return False
            elif isinstance(orig_val, Tensor) and isinstance(recon_val, Tensor):
                # Move both to CPU for comparison to handle device mismatches
                if not torch.equal(orig_val.cpu(), recon_val.cpu()):
                    print(f"Tensor mismatch at {current_path}: shapes={orig_val.shape} vs {recon_val.shape}")
                    return False
            elif orig_val is None and recon_val is None:
                pass  # Both are None, continue
            elif isinstance(orig_val, (int, float, str, bool)) and isinstance(recon_val, (int, float, str, bool)):
                if orig_val != recon_val:
                    print(f"Value mismatch at {current_path}: {orig_val} vs {recon_val}")
                    return False
            elif type(orig_val) is not type(recon_val):
                print(f"Type mismatch at {current_path}: {type(orig_val)} vs {type(recon_val)}")
                return False

        return True
    print(f"Type mismatch at {path or 'root'}: {type(original)} vs {type(reconstructed)}")
    return False


def convert_checkpoint(file: Path | str) -> None:
    tensors: dict[str, object] = cast(dict[str, object], torch.load(base / file))
    assert isinstance(tensors, dict), f"Expected a dict, got {type(tensors)} from {file}"
    new_file = (base / file).with_suffix(".safetensors")
    if new_file.exists():
        print(f"File {new_file} already exists, skipping conversion.")
        return

    try:
        flat_tensors, metadata = flatten_tensors(tensors)
        safetensors.torch.save_file(flat_tensors, new_file, metadata=metadata)

        # Verify round-trip conversion
        loaded_tensors = safetensors.torch.load_file(new_file)
        loaded_metadata = safetensors.torch.load_file(new_file, device="cpu")
        # Get metadata from the file header
        with Path(new_file).open("rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header_bytes = f.read(header_size)
            import json

            header = json.loads(header_bytes)
            loaded_metadata = header.get("__metadata__", {})

        reconstructed = unflatten_tensors(loaded_tensors, loaded_metadata)
        if compare_structures(tensors, reconstructed):
            print(f"✓ Successfully converted {file} to {new_file.name} (structure verified)")
        else:
            print(f"⚠ Converted {file} to {new_file.name} but structure mismatch detected!")
    except Exception as e:
        new_file.unlink(missing_ok=True)
        print(f"ERROR: Failed to convert file `{file}`:", e)
        raise


# convert_checkpoint("wav2vec2bert_stats.pt")
# convert_checkpoint("feat1.pt")
# convert_checkpoint("feat2.pt")
# convert_checkpoint("gpt.pth")
convert_checkpoint("s2mel.pth")
