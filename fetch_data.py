"""Pull single cases out of the TotalSegmentator zips on Zenodo without downloading
the 24-37 GB archives.

The zip central directory is read once (remotezip, HTTP Range) and cached as an index.
All files of one case sit next to each other inside the zip, so each case is fetched
with ONE Range request and unpacked locally (Zenodo rate-limits many small requests).

  python fetch_data.py meta              -> data/meta_v{2,3}.csv, data/index_v{2,3}.json
  python fetch_data.py cases v3 s1234 .. -> data/v3/s1234/{ct.nii.gz, segmentations/*.nii.gz}
"""
import json, os, struct, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor

import requests
from remotezip import RemoteZip

URL = {
    "v2": "https://zenodo.org/api/records/10047292/files/Totalsegmentator_dataset_v201.zip/content",
    "v3": "https://zenodo.org/api/records/22688904/files/Totalsegmentator_dataset_v300.zip/content",
}
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def meta(vers):
    for ver in vers:
        with RemoteZip(URL[ver], initial_buffer_size=1 << 20) as z:
            infos = z.infolist()
            m = [i.filename for i in infos if i.filename.endswith("meta.csv")][0]
            with open(os.path.join(ROOT, f"meta_{ver}.csv"), "wb") as f:
                f.write(z.read(m))
            index = [[i.filename, i.header_offset, i.compress_size, i.compress_type]
                     for i in infos if not i.filename.endswith("/")]
            with open(os.path.join(ROOT, f"index_{ver}.json"), "w") as f:
                json.dump(index, f)
            print(ver, len(index), "files indexed", flush=True)


def get_range(url, start, end, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120, stream=True)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                print(f"  429, waiting {wait}s", flush=True); time.sleep(wait); continue
            r.raise_for_status()
            assert r.status_code == 206, r.status_code
            buf = bytearray()
            for chunk in r.iter_content(1 << 20):
                buf += chunk
            if len(buf) == end - start + 1:
                return bytes(buf)
            print(f"  short read {len(buf)}, retry", flush=True)
        except (requests.RequestException, AssertionError) as e:
            print(f"  error {e}, retry", flush=True); time.sleep(5 * (k + 1))
    raise RuntimeError("range download failed")


def get_range_parallel(url, start, end, parts=16):
    """~0.2 MB/s per connection from here, so split the range across many connections."""
    step = max(1 << 20, -(-(end - start + 1) // parts))
    chunks = [(s, min(end, s + step - 1)) for s in range(start, end + 1, step)]
    with ThreadPoolExecutor(len(chunks)) as ex:
        return b"".join(ex.map(lambda c: get_range(url, *c), chunks))


def resolve(url):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
    return r.url


def cases(ver, ids):
    index = json.load(open(os.path.join(ROOT, f"index_{ver}.json")))
    offsets = sorted(i[1] for i in index)
    url = resolve(URL[ver])
    print('storage url:', url[:80], flush=True)
    for cid in ids:
        t = time.time()
        members = [i for i in index if f"/{cid}/" in "/" + i[0]]
        if all(os.path.exists(os.path.join(ROOT, ver, m[0][m[0].index(cid):])) for m in members):
            print(f"{ver}/{cid}: already here", flush=True); continue
        start = min(m[1] for m in members)
        last = max(m[1] for m in members)
        nxt = next((o for o in offsets if o > last), None)
        end = (nxt - 1) if nxt else last + max(m[2] for m in members) + 65536
        blob = get_range_parallel(url, start, end)
        size = 0
        for name, off, csize, ctype in members:
            p = off - start
            sig, = struct.unpack("<I", blob[p:p + 4])
            assert sig == 0x04034B50, f"bad local header for {name}"
            nlen, xlen = struct.unpack("<HH", blob[p + 26:p + 30])
            raw = blob[p + 30 + nlen + xlen: p + 30 + nlen + xlen + csize]
            data = raw if ctype == 0 else zlib.decompress(raw, -15)
            dst = os.path.join(ROOT, ver, name[name.index(cid):])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
            size += len(data)
        dt = time.time() - t
        print(f"{ver}/{cid}: {len(members)} files, {len(blob)/1e6:.0f} MB in {dt:.0f} s "
              f"({len(blob)/1e6/max(dt,1e-3):.1f} MB/s)", flush=True)


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    if sys.argv[1] == "meta":
        meta(sys.argv[2:] or list(URL))
    else:
        cases(sys.argv[2], sys.argv[3:])
