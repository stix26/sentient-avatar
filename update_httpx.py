import re
import subprocess
from pathlib import Path

REQ_FILE = Path('requirements.txt')
HTTPX_CONFLICT_VERSION = '0.26.0'
HTTPX_FIXED_RANGE = '>=0.28.1,<1.0.0'

def main():
    if not REQ_FILE.exists():
        print('requirements.txt not found.')
        return

    lines = REQ_FILE.read_text().splitlines()
    httpx_idx = None
    autogen_present = any('autogen' in line and not line.strip().startswith('#') for line in lines)

    for i, line in enumerate(lines):
        if line.strip().startswith('httpx=='):
            httpx_idx = i
            break

    if httpx_idx is not None:
        version = lines[httpx_idx].split('==')[1]
        if version == HTTPX_CONFLICT_VERSION and autogen_present:
            print(
                f'Warning: httpx=={HTTPX_CONFLICT_VERSION} conflicts with autogen. '
                f'Updating to httpx{HTTPX_FIXED_RANGE}.'
            )
            lines[httpx_idx] = f'httpx{HTTPX_FIXED_RANGE}'
            REQ_FILE.write_text('\n'.join(lines) + '\n')
        else:
            print('No conflicting httpx version found.')
    else:
        print('httpx not pinned, nothing to update.')

    # Bonus: install packages
    try:
        subprocess.run(['pip', 'install', '-r', str(REQ_FILE)], check=True)
    except subprocess.CalledProcessError as exc:
        print('pip install failed:', exc)

if __name__ == '__main__':
    main()
