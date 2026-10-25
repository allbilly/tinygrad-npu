"""Device-free configuration and source identity for the projection experiment."""
import hashlib
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRECTORY = Path(__file__).resolve().parent
GEOMETRIES = {f'm{m}_n{n}_k{k}': {'rows': m, 'columns': n, 'depth': k}
              for m, n, k in ((64, 2048, 2048), (64, 6144, 2048), (64, 12288, 2048), (64, 2048, 4096))}
MEMORY_RESERVE_BYTES = 2 * 1024**3
STARTUP_HEADROOM_BYTES = 512 * 1024**2


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def source_map(root: Path = SOURCE_ROOT) -> dict[str, str]:
  files = sorted(p for p in (root / 'tinygrad').rglob('*') if p.is_file() and p.suffix in ('.py', '.c', '.h'))
  files += sorted((root / 'extra/rockchip/shared_prefill').glob('*.py'))
  return {str(path.relative_to(root)): sha256(path) for path in files}


def check_memory(startup: bool = False) -> int:
  available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))
  minimum = MEMORY_RESERVE_BYTES + (STARTUP_HEADROOM_BYTES if startup else 0)
  if available < minimum:
    raise MemoryError(f'Available RAM {available} is below the required {minimum} bytes')
  return available
