"""
zenodo_deposit.py — publish the CD-BAN code-only archive to Zenodo.

Uses the CURRENT InvenioRDM Records API (Zenodo retired the old
/api/deposit/depositions endpoints). Self-contained: it builds the code-only
tarball from the committed git tree (so confidential drafts / agent state /
.git / data / weights / figures are excluded), transforms .zenodo.json into
the new metadata format, uploads, publishes, and prints the DOI.

Verified flow against zenodo.org:
  POST /api/records                            -> create draft
  POST /api/records/{id}/draft/files           -> register file (returns dict.entries[])
  PUT  <entry.links.content>                   -> stream bytes (local transfer)
  POST /api/records/{id}/draft/files/{k}/commit-> commit
  POST /api/records/{id}/draft/actions/publish -> publish
  GET  /api/records/{id}                       -> read links.doi

Prerequisite: a Zenodo personal access token with deposit:write +
deposit:actions scopes (https://zenodo.org/settings/tokens). Do NOT hardcode
it here — pass it via the ZENODO_TOKEN environment variable.

Usage:
  export ZENODO_TOKEN="your_token"
  python zenodo_deposit.py                 # production (zenodo.org)
  python zenodo_deposit.py --sandbox       # test on sandbox.zenodo.org (separate account)
  python zenodo_deposit.py --dry-run       # build + validate the tarball, do NOT upload

After publishing, replace [[ZENODO_DOI]] in the manuscript Code Availability
statement with the printed DOI.
"""
import argparse, datetime, json, os, re, subprocess, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARBALL = os.path.join(HERE, "CD-BAN_code_only.tar.gz")
META = os.path.join(HERE, ".zenodo.json")
DOI_OUT = os.path.join(HERE, "zenodo_doi.txt")

# Code-only file set: source + config + env + docs. Excludes data (csv),
# weights (pth), figures (png/svg/pdf), logs, and gitignored drafts.
CODE_FILTER = re.compile(
    r"\.(py|sh|yaml|cff)$|^requirements\.txt$|^README\.md$|^ALGORITHM\.md$"
    r"|^LICENSE$|^\.zenodo\.json$|^\.gitignore$")

# Extensions / names that must NEVER be in a code-only release.
FORBIDDEN_EXT = (".csv", ".pth", ".png", ".svg", ".pdf", ".pdb", ".pml", ".npy", ".log")
FORBIDDEN_NAME = ("文章", "response_letter", "SUBMISSION_CHECKLIST")


def build_code_only_tarball():
    out = subprocess.check_output(["git", "ls-files"], cwd=HERE).decode().splitlines()
    files = sorted(f for f in out if CODE_FILTER.search(f))
    if not files:
        sys.exit("No code files matched the filter — is this a git checkout?")
    print(f"[1/5] Building code-only tarball from git HEAD: {len(files)} files")
    if os.path.exists(TARBALL):
        os.remove(TARBALL)
    with tarfile.open(TARBALL, "w:gz") as tar:
        for f in files:
            tar.add(os.path.join(HERE, f), arcname=f"CD-BAN/{f}")
    size_mb = os.path.getsize(TARBALL) / 1e6
    print(f"      {size_mb:.1f} MB -> {os.path.basename(TARBALL)}")
    with tarfile.open(TARBALL) as tar:
        names = tar.getnames()
    bad = [n for n in names
           if n.endswith(FORBIDDEN_EXT)
           or any(k in n for k in FORBIDDEN_NAME)]
    if bad:
        sys.exit(f"SAFETY: forbidden files in tarball: {bad[:10]}")
    print("      safety check: CLEAN (no data/weights/figures/manuscript)")
    return os.path.basename(TARBALL)


def new_format_metadata():
    old = json.load(open(META))
    creators = []
    for c in old["creators"]:
        name = c["name"]
        po = {"name": name, "type": "personal"}
        if "," in name:
            fam, given = name.split(",", 1)
            po["family_name"] = fam.strip()
            po["given_name"] = given.strip()
        entry = {"person_or_org": po, "role": {"id": "other"}}
        if c.get("affiliation"):
            entry["affiliations"] = [{"name": c["affiliation"]}]
        if c.get("orcid"):
            po["identifiers"] = [{"scheme": "orcid", "identifier": c["orcid"]}]
        creators.append(entry)
    desc = old["description"]
    if "github.com/CHIHX12/CD-BAN" not in desc:
        desc += " Code: https://github.com/CHIHX12/CD-BAN"
    return {
        "title": old["title"],
        "creators": creators,
        "description": desc,
        "publisher": "Royal Society of Chemistry",   # required for DOI registration
        "publication_date": datetime.date.today().isoformat(),
        "resource_type": {"id": "software"},
        "keywords": old.get("keywords", []),
        "license": {"id": "mit"},
        "language": old.get("language", "eng"),
        "notes": old.get("notes", ""),
        # related_identifiers omitted: the new API validates relation_type
        # strictly, and the GitHub link is already in the description.
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--dry-run", action="store_true", help="build+validate tarball only")
    args = ap.parse_args()

    fn = build_code_only_tarball()

    if args.dry_run:
        print("\nDRY RUN: tarball built and validated. Not uploading.")
        return

    import requests
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN (personal access token, "
                 "deposit:write + deposit:actions).")
    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    H = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
         "Accept": "application/vnd.inveniordm.v1+json"}

    def chk(r, step):
        if r.status_code >= 400:
            print(f"!! {step}: HTTP {r.status_code}\n{r.text[:800]}")
            sys.exit(1)
        return r.json()

    print("[2/5] create draft ...")
    r = chk(requests.post(f"{base}/records", headers=H,
                          json={"access": {"record": "public", "files": "public"},
                                "files": {"enabled": True},
                                "metadata": new_format_metadata()}), "create draft")
    rid = r["id"]
    print(f"      draft id = {rid}")

    print("[3/5] register + upload file ...")
    reg = chk(requests.post(f"{base}/records/{rid}/draft/files", headers=H,
                            json=[{"key": fn}]), "register file")
    content = reg["entries"][0]["links"]["content"]
    with open(TARBALL, "rb") as fh:
        chk(requests.put(content, data=fh, headers={"Authorization": f"Bearer {token}"},
                         timeout=600), "upload")
    chk(requests.post(f"{base}/records/{rid}/draft/files/{fn}/commit", headers=H), "commit")
    print("      uploaded + committed.")

    print("[4/5] publish ...")
    chk(requests.post(f"{base}/records/{rid}/draft/actions/publish", headers=H), "publish")

    print("[5/5] read DOI ...")
    pub = requests.get(f"{base}/records/{rid}", headers=H).json()
    doi = pub.get("links", {}).get("doi", "").replace("https://doi.org/", "")
    landing = pub.get("links", {}).get("self_html") or f"https://zenodo.org/records/{rid}"
    print("\n" + "=" * 55)
    print(f"  PUBLISHED  DOI: {doi}")
    print(f"  Landing:   {landing}")
    print("=" * 55)
    if doi:
        open(DOI_OUT, "w").write(doi)
        print(f"\n  Saved to {os.path.basename(DOI_OUT)}.")
        print("  Now replace [[ZENODO_DOI]] in the manuscript Code Availability")
        print(f"  statement with: {doi}")


if __name__ == "__main__":
    main()
