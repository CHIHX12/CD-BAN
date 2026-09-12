"""
zenodo_deposit.py
=================
Deposit the CD-BAN release to Zenodo and obtain the DOI required by the
Digital Discovery Data Availability Statement.

Prerequisites:
  1. A Zenodo account (https://zenodo.org)
  2. A personal access token:
       - Go to https://zenodo.org/settings/tokens  (or GitHub OAuth for the
         sandbox), create a token with "deposit:write" scope, OR
       - Use the sandbox for testing: https://sandbox.zenodo.org

Usage:
  export ZENODO_TOKEN="your_token_here"
  python zenodo_deposit.py                 # production (zenodo.org)
  python zenodo_deposit.py --sandbox       # test on sandbox.zenodo.org
  python zenodo_deposit.py --draft         # create draft + upload, do NOT publish

After publishing, the script prints the DOI. Paste it into the manuscript
placeholder  [[ZENODO_DOI]]  (search for it) in the Code Availability statement.

The release tarball is built automatically from the committed git tree so that
confidential drafts / agent state / .git are excluded (they are gitignored).
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TARBALL = os.path.join(HERE, "CD-BAN_release.tar.gz")
META = os.path.join(HERE, ".zenodo.json")

def build_tarball():
    """Create the release tarball from the committed tree (excludes gitignored files)."""
    print(f"[1/4] Building release tarball from git HEAD -> {os.path.basename(TARBALL)}")
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "--prefix=CD-BAN/", "-o", TARBALL, "HEAD"],
        check=True, cwd=HERE,
    )
    size_mb = os.path.getsize(TARBALL) / 1e6
    print(f"      {size_mb:.1f} MB")

def load_meta():
    with open(META) as f:
        meta = json.load(f)
    print(f"[2/4] Loaded metadata: {meta['title'][:60]}...")
    print(f"      creators: {', '.join(c['name'] for c in meta['creators'])}")
    return meta

def zenodo_request(method, url, token, data=None, files=None, headers=None):
    import requests
    h = {"Authorization": f"Bearer {token}"}
    if headers: h.update(headers)
    r = requests.request(method, url, auth=None, headers=h, json=data, files=files, timeout=300)
    if r.status_code >= 400:
        print(f"      !! HTTP {r.status_code}: {r.text[:400]}")
        r.raise_for_status()
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--draft", action="store_true", help="do not publish (create draft + upload only)")
    args = ap.parse_args()

    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN env var (personal access token with deposit:write scope).")

    build_tarball()
    meta = load_meta()

    # 3. create deposit
    print(f"[3/4] Creating deposit on {base} ...")
    r = zenodo_request("POST", f"{base}/api/deposit/deposits", token, data={"metadata": meta})
    dep = r.json()
    dep_id = dep["id"]
    print(f"      deposit id = {dep_id}")

    # 4. upload file
    print("[4/4] Uploading tarball ...")
    with open(TARBALL, "rb") as fh:
        zenodo_request("PUT", f"{base}/api/deposit/deposits/{dep_id}/files/{os.path.basename(TARBALL)}",
                       token, files={"file": (os.path.basename(TARBALL), fh)})
    print("      uploaded.")

    if args.draft:
        print(f"\nDRAFT mode: not published. Deposit URL: {base}/record/{dep_id}")
        print("Publish later via the web UI or re-run without --draft.")
        return

    # publish
    print("Publishing ...")
    r = zenodo_request("POST", f"{base}/api/deposit/deposits/{dep_id}/actions/publish", token)
    pub = r.json()
    doi = pub.get("doi")
    landing = pub.get("links", {}).get("self") or f"{base}/record/{dep_id}"
    print("\n" + "=" * 60)
    print(f"  PUBLISHED  DOI: {doi}")
    print(f"  Landing:   {landing}")
    print(f"  Cite as:   https://doi.org/{doi}")
    print("=" * 60)
    print(f"\n  NOW: open the manuscript and replace  [[ZENODO_DOI]]  with  {doi}")
    print(f"  (it appears in the Code Availability statement).")

if __name__ == "__main__":
    main()
